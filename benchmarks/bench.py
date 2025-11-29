"""
Metrics tested:
- WER (Word Error Rate) on test-clean and test-other
- Latency (p50/p95/p99) with batch_size=1 
- Throughput (utterances/sec) with batch_size=16
- GPU Memory usage
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from moe.patch import inject_moe

from utils import (
    Timer,
    get_wer,
    get_latency,
    get_gpu_memory_mb,
    reset,
    normalize_text,
    extract_output
)

FLEURS_LANGUAGES = {
    # Finetuning languages
    "es_419": ("spa", "romance", "train"),
    "pt_br": ("por", "romance", "train"),
    "pl_pl": ("pol", "slavic", "train"),
    "cs_cz": ("ces", "slavic", "train"),
    # Held-out languages
    "it_it": ("ita", "romance", "heldout"),
    "fr_fr": ("fra", "romance", "heldout"),
    "sk_sk": ("slk", "slavic", "heldout"),
    "hr_hr": ("hrv", "slavic", "heldout"),
}


class BaselineBenchmark:
    """Benchmark suite for OWSM-CTC baseline model."""
    
    def __init__(
        self,
        model_name: str = "espnet/owsm_ctc_v4_1B",
        model_path: str = None,
        device: str = "cuda",
        data_dir: str = "data/librispeech",
        results_dir: str = "results/baseline",
        use_flash_attn: bool = True,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.use_flash_attn = use_flash_attn
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # DOES BF16 WORK?
        print(f"Loading OWSM-CTC model: {model_name}")
        if model_path:
            print(f"Loading fine-tuned model from: {model_path}")
            self._load_finetuned_model(model_path)
        else:
            print(f"Loading baseline OWSM-CTC model: {model_name}")
            self._load_baseline_model(model_name)
        
        print(f"Model loaded")
        print(f"Flash Attention: {use_flash_attn}")
        print(f"Device: {device}")
    
    def _load_baseline_model(self, model_name: str):
        """Load baseline model from HuggingFace."""
        # TODO: ADJUST FOR FLASH ATTENTION LATER
        if self.use_flash_attn:
            self.model = Speech2TextGreedySearch.from_pretrained(
                model_name,
                device=self.device,
                lang_sym="<eng>",
                task_sym="<asr>",
            )
        else:
            self.model = Speech2TextGreedySearch.from_pretrained(
                model_name,
                device=self.device,
                lang_sym="<eng>",
                task_sym="<asr>",
            )
        
        '''# TEMPORARY: Inject MoE into last 9 layers for testing
        print("\n" + "="*60)
        print("INJECTING MoE INTO LAST 9 LAYERS")
        print("="*60)
        encoder = self.model.s2t_model.encoder
        num_layers = len(encoder.encoders)
        last_9_layers = list(range(num_layers - 9, num_layers))
        print(f"Total layers: {num_layers}")
        print(f"Injecting MoE into layers: {last_9_layers}")
        
        inject_moe(
            encoder,
            n_experts=8,
            top_k=1,
            capacity_factor=1.25,
            noisy_gate_std=1.0,
            use_noisy_gating=False,  # inference mode
            layers=last_9_layers,
            replace_macaron=True,
            init_from_pretrained=True,
            init_noise_std=0.0,  # inference mode
            verbose=True,
        )
        
        # Move MoE layers to the correct device
        print(f"Moving model to device: {self.device}")
        self.model.s2t_model.to(self.device)
        self.model.s2t_model.eval()
        print("="*60 + "\n")'''

    def _load_finetuned_model(self, model_path: str):
        """Load fine-tuned MoE model from checkpoint."""
        from moe.moe_ffn import MoEPositionwiseFFN
        
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        print(f"Loading MoE fine-tuned model from: {model_path}")
        
        print("Step 1: Loading base OWSM-CTC model...")
        self.model = Speech2TextGreedySearch.from_pretrained(
            self.model_name,
            device=self.device, 
            lang_sym="<eng>",
            task_sym="<asr>",
        )
        
        # We inject MoE layers after loading the base model
        print("Step 2: Injecting MoE layers into last 9 encoder layers...")
        encoder = self.model.s2t_model.encoder
        num_layers = len(encoder.encoders)
        target_layers = list(range(num_layers - 9, num_layers))
        
        inject_moe(
            encoder,
            n_experts=8,
            top_k=1,
            capacity_factor=2.5,  # Match training config
            noisy_gate_std=0.25,
            use_noisy_gating=False,  # Disable noise for inference
            layers=target_layers,
            replace_macaron=True,
            init_from_pretrained=False,  # We'll load weights from checkpoint
            init_noise_std=0.0,
            verbose=True,
        )
        
        # Then we can load checkpoint weights
        print(f"Step 3: Loading checkpoint weights...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # ESPnet saves model state_dict under different keys
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load the state dict
        missing, unexpected = self.model.s2t_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")
        
        # Step 4: Move ENTIRE model to device and set eval mode
        print(f"Step 4: Moving model to {self.device}...")
        # Move the s2t_model
        self.model.s2t_model.to(self.device)
        self.model.s2t_model.eval()
        # Update the device attribute used by Speech2TextGreedySearch
        self.model.device = self.device
        if hasattr(self.model, 'beam_search') and self.model.beam_search is not None:
            self.model.beam_search.to(self.device)
        moe_count = sum(1 for m in self.model.s2t_model.encoder.modules() 
                        if isinstance(m, MoEPositionwiseFFN))
        print(f"Verified: {moe_count} MoE layers loaded")
        print("MoE model loaded successfully!")
        
    def load_librispeech_subset(
        self, 
        split: str, 
        max_samples: int = None
    ) -> List[Tuple[torch.Tensor, int, str]]:
        """
        Load LibriSpeech dataset.
        """
        print(f"\nLoading {split}...")
        
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(self.data_dir),
            url=split,
            download=False,
        )
        
        samples = []
        for i, (waveform, sample_rate, transcript, _, _, _) in enumerate(
            tqdm(dataset, desc=f"Loading {split}")
        ):
            samples.append((waveform[0].numpy(), sample_rate, transcript))
            if max_samples and i + 1 >= max_samples:
                break
        
        print(f"Loaded {len(samples)} samples from {split}")
        return samples
    
    def load_fleurs_subset(
        self,
        fleurs_code: str,
        split: str = "test",
        max_samples: int = None,
        cache_dir: str = "data/fleurs",
    ) -> List[Tuple[np.ndarray, int, str]]:
        """
        Load FLEURS dataset for a specific language. 
        """
        from datasets import load_dataset
        
        if fleurs_code not in FLEURS_LANGUAGES:
            raise ValueError(f"Unknown FLEURS language: {fleurs_code}")
        
        iso3, family, role = FLEURS_LANGUAGES[fleurs_code]
        print(f"\nLoading FLEURS {fleurs_code} ({iso3}) - {split}...")
        
        dataset = load_dataset(
            "google/fleurs",
            fleurs_code,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        samples = []
        for i, sample in enumerate(tqdm(dataset, desc=f"Loading {fleurs_code}")):
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            transcript = sample["transcription"]
            samples.append((audio, sr, transcript))
            
            if max_samples and i + 1 >= max_samples:
                break
        
        print(f"Loaded {len(samples)} samples from {fleurs_code} ({split})")
        return samples
    
    def test_wer(
        self, 
        split: str = "test-clean",
        max_samples: int = None
    ) -> Dict:
        """
        Test Word Error Rate on a dataset split.
        """
        print(f"WER Test: {split}")
        samples = self.load_librispeech_subset(split, max_samples)
        reset()
        # Run inference
        ref = []
        hyp = []
        with torch.inference_mode():
            for i, (audio, sr, ref_text) in enumerate(tqdm(samples, desc="Inference")):
                result = self.model(audio)

                hyp_text = extract_output(result)
                ref_text_norm = normalize_text(ref_text)
                hyp_text_norm = normalize_text(hyp_text)

                #if i < 20:
                 #   print(f"\n=== DEBUG FIRST SAMPLE ===")
                  #  print(f"Reference (raw): '{ref_text}'")
                   # print(f"Hypothesis (raw): '{hyp_text}'")
                    #print(f"Hypothesis (norm): '{hyp_text_norm}'")
                    #print(f"Match? {ref_text_norm == hyp_text_norm}")
                    #print(f"==========================\n")
                
                ref.append(ref_text_norm)
                hyp.append(hyp_text_norm)
        # Calculate WER
        wer_score = get_wer(ref, hyp)
        memory_mb = get_gpu_memory_mb()
        results = {
            "split": split,
            "num_samples": len(samples),
            "wer_percent": round(wer_score, 2),
            "peak_memory_mb": round(memory_mb, 2),
        }
        
        print(f"\n{'='*60}")
        print(f"WER Results: {split}")
        print(f"{'='*60}")
        print(f"Samples: {results['num_samples']}")
        print(f"WER: {results['wer_percent']:.2f}%")
        print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
        print(f"{'='*60}")
        return results
    
    def test_wer_fleurs(
        self,
        fleurs_code: str,
        split: str = "test",
        max_samples: int = None,
        cache_dir: str = "data/fleurs",
    ) -> Dict:
        """
        Test Word Error Rate on a FLEURS language. 
        """
        if fleurs_code not in FLEURS_LANGUAGES:
            raise ValueError(f"Unknown FLEURS language: {fleurs_code}")
        
        iso3, family, role = FLEURS_LANGUAGES[fleurs_code]
        
        print(f"\n{'='*60}")
        print(f"FLEURS WER Test: {fleurs_code} ({iso3}) - {family}")
        print(f"{'='*60}")
        
        # Update model's language symbol for this language
        self.model.lang_sym = f"<{iso3}>"
        
        samples = self.load_fleurs_subset(fleurs_code, split, max_samples, cache_dir)
        reset()
        
        ref = []
        hyp = []
        
        with torch.inference_mode():
            for i, (audio, sr, ref_text) in enumerate(tqdm(samples, desc="Inference")):
                result = self.model(audio)
                
                hyp_text = extract_output(result)
                ref_text_norm = normalize_text(ref_text)
                hyp_text_norm = normalize_text(hyp_text)
                
                ref.append(ref_text_norm)
                hyp.append(hyp_text_norm)
        
        wer_score = get_wer(ref, hyp)
        memory_mb = get_gpu_memory_mb()
        
        results = {
            "fleurs_code": fleurs_code,
            "iso3": iso3,
            "family": family,
            "role": role,
            "split": split,
            "num_samples": len(samples),
            "wer_percent": round(wer_score, 2),
            "peak_memory_mb": round(memory_mb, 2),
        }
        
        print(f"\n{'='*60}")
        print(f"FLEURS WER Results: {fleurs_code} ({iso3})")
        print(f"{'='*60}")
        print(f"Family: {family} | Role: {role}")
        print(f"Samples: {results['num_samples']}")
        print(f"WER: {results['wer_percent']:.2f}%")
        print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
        print(f"{'='*60}")
        
        return results
    

    def run_fleurs_baseline(
        self,
        languages: List[str] = None,
        split: str = "test",
        max_samples: int = None,
        cache_dir: str = "data/fleurs",
    ) -> Dict:
        """
        Run WER evaluation on multiple FLEURS languages. 
        """
        from datetime import datetime
        
        if languages is None:
            languages = list(FLEURS_LANGUAGES.keys())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "timestamp": timestamp,
            "model": self.model_name,
            "model_path": self.model_path if self.model_path else "baseline",
            "split": split,
            "languages": {},
        }
        
        for fleurs_code in languages:
            results["languages"][fleurs_code] = self.test_wer_fleurs(
                fleurs_code=fleurs_code,
                split=split,
                max_samples=max_samples,
                cache_dir=cache_dir,
            )
        
        # Summary by family
        print(f"\n{'='*60}")
        print("FLEURS BASELINE SUMMARY")
        print(f"{'='*60}")
        
        for family in ["romance", "slavic"]:
            for role in ["train", "heldout"]:
                langs = [k for k, v in FLEURS_LANGUAGES.items() 
                        if v[1] == family and v[2] == role and k in results["languages"]]
                if langs:
                    avg_wer = np.mean([results["languages"][l]["wer_percent"] for l in langs])
                    print(f"{family.capitalize()} ({role}): {avg_wer:.2f}% WER")
                    for l in langs:
                        iso3 = FLEURS_LANGUAGES[l][0]
                        wer_val = results["languages"][l]["wer_percent"]
                        print(f"  - {l} ({iso3}): {wer_val:.2f}%")
        
        # Save results
        model_type = "finetuned" if self.model_path else "baseline"
        results_file = self.results_dir / f"fleurs_{model_type}_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return results
    
    def test_latency(
        self,
        split: str = "test-clean",
        num_samples: int = 1000,
        warmup: int = 50,
    ) -> Dict:
        """
        Test per-utterance latency with batch_size=1.
        """
        print(f"Latency Test: {split}")
        samples = self.load_librispeech_subset(split, num_samples + warmup)
        reset()
        latencies = []
        
        with torch.inference_mode():
            for i, (audio, sr, _) in enumerate(
                tqdm(samples, desc="Measuring latency")
            ):
                timer = Timer(device=self.device)
                with timer:
                    _ = self.model(audio)
                
                if i >= warmup:
                    latencies.append(timer.elapsed)
        
        latency_stats = get_latency(latencies)
        memory_mb = get_gpu_memory_mb()
        results = {
            "split": split,
            "num_samples": len(latencies),
            "warmup_samples": warmup,
            **latency_stats,
            "peak_memory_mb": round(memory_mb, 2),
        }
        
        print(f"\n{'='*60}")
        print(f"Latency Results: {split}")
        print(f"{'='*60}")
        print(f"Samples: {results['num_samples']}")
        print(f"Mean: {results['mean_ms']:.1f} ms")
        print(f"Std: {results['std_ms']:.1f} ms")
        print(f"p50: {results['p50_ms']:.1f} ms")
        print(f"p95: {results['p95_ms']:.1f} ms")
        print(f"p99: {results['p99_ms']:.1f} ms")
        print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
        print(f"{'='*60}")
        
        return results
    
    def test_throughput(
        self,
        split: str = "test-clean",
        num_samples: int = 500,
        warmup: int = 50,
    ) -> Dict:
        """
        Test throughput (utterances/second) (sequential processing).
        """
        print(f"\n{'='*60}")
        print(f"Throughput Test: {split}")
        print(f"{'='*60}")
        samples = self.load_librispeech_subset(split, num_samples + warmup)
        reset()
        # Warmup
        with torch.inference_mode():
            for audio, sr, _ in samples[:warmup]:
                _ = self.model(audio)
        
        test_samples = samples[warmup:]
        timer = Timer(device=self.device)
        with torch.inference_mode(), timer:
            for audio, sr, _ in test_samples:
                _ = self.model(audio)
        total_time = timer.elapsed
        throughput = len(test_samples) / total_time
        memory_mb = get_gpu_memory_mb()
        results = {
            "split": split,
            "num_samples": len(test_samples),
            "warmup_samples": warmup,
            "total_time_sec": round(total_time, 2),
            "throughput_utt_per_sec": round(throughput, 2),
            "time_per_utt_ms": round((total_time / len(test_samples)) * 1000, 2),
            "peak_memory_mb": round(memory_mb, 2),
        }
        
        print(f"\n{'='*60}")
        print(f"Throughput Results: {split}")
        print(f"{'='*60}")
        print(f"Samples: {results['num_samples']}")
        print(f"Total Time: {results['total_time_sec']:.1f} sec")
        print(f"Throughput: {results['throughput_utt_per_sec']:.2f} utt/sec")
        print(f"Time per utt: {results['time_per_utt_ms']:.1f} ms")
        print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
        print(f"{'='*60}")
        return results
    
    def run_full(self) -> Dict:
        """
        Run all tests
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "model": self.model_name,
            "model_path": self.model_path if self.model_path else "baseline",
            "device": self.device,
            "use_flash_attn": self.use_flash_attn,
            "wer": {},
            "latency": {},
            "throughput": {},
        }
        for split in ["test-clean", "test-other"]:
            results["wer"][split] = self.test_wer(split=split)
        results["latency"]["test-clean"] = self.test_latency(
            split="test-clean",
            num_samples=500,
        )
        results["throughput"]["test-clean"] = self.test_throughput(
            split="test-clean",
            num_samples=500,
        )
        model_type = "finetuned" if self.model_path else "baseline"
        results_file = self.results_dir / f"{model_type}_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OWSM-CTC"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="espnet/owsm_ctc_v4_1B",
        help="Model name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (overrides model name if provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/librispeech",
        help="LibriSpeech data directory",
    )
    parser.add_argument(
        "--fleurs_dir",
        type=str,
        default="data/fleurs",
        help="FLEURS cache directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/baseline",
        help="Results output directory",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable Flash Attention",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["wer", "latency", "throughput", "all", "perf", "fleurs"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--fleurs_langs",
        type=str,
        nargs="+",
        default=None,
        help="FLEURS language codes to test (e.g., es_419 pl_pl)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per language (for quick testing)",
    )
    
    args = parser.parse_args()
    
    benchmark = BaselineBenchmark(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_flash_attn=not args.no_flash_attn,
    )
    
    if args.test == "all":
        results = benchmark.run_full()
    elif args.test == "wer":
        results = {}
        for split in ["test-clean", "test-other"]:
            results[split] = benchmark.test_wer(split=split)
    elif args.test == "latency":
        results = benchmark.test_latency()
    elif args.test == "throughput":
        results = benchmark.test_throughput()
    elif args.test == "perf":
        results = {}
        results["latency"] = benchmark.test_latency()
        results["throughput"] = benchmark.test_throughput()
    elif args.test == "fleurs":
        results = benchmark.run_fleurs_baseline(
            languages=args.fleurs_langs,
            split="test",
            max_samples=args.max_samples,
            cache_dir=args.fleurs_dir,
        )
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()