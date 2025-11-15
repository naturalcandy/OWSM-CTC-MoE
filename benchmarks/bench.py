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

from utils import (
    Timer,
    get_wer,
    get_latency,
    get_gpu_memory_mb,
    reset,
    normalize_text,
    extract_output
)


class BaselineBenchmark:
    """Benchmark suite for OWSM-CTC baseline model."""
    
    def __init__(
        self,
        model_name: str = "espnet/owsm_ctc_v4_1B",
        device: str = "cuda",
        data_dir: str = "data/librispeech",
        results_dir: str = "results/baseline",
        use_flash_attn: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.use_flash_attn = use_flash_attn
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # DOES BF16 WORK?
        print(f"Loading OWSM-CTC model: {model_name}")
        
        if use_flash_attn:
            self.model = Speech2TextGreedySearch.from_pretrained(
                model_name,
                device=device,
                lang_sym="<eng>",
                task_sym="<asr>",
                use_flash_attn=True,
                dtype="float16",
            )
        else:
            self.model = Speech2TextGreedySearch.from_pretrained(
                model_name,
                device=device,
                lang_sym="<eng>",
                task_sym="<asr>",
            )
        print(f"Model loaded")
        print(f"Flash Attention: {use_flash_attn}")
        print(f"Device: {device}")
        
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
            transcript_norm = normalize_text(transcript)
            samples.append((waveform[0].numpy(), sample_rate, transcript_norm))
            
            if max_samples and i + 1 >= max_samples:
                break
        
        print(f"Loaded {len(samples)} samples from {split}")
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
                hyp_text_norm = normalize_text(hyp_text)

                if i == 0:
                    print(f"\n=== DEBUG FIRST SAMPLE ===")
                    print(f"Reference (raw): '{ref_text}'")
                    print(f"Hypothesis (raw): '{hyp_text}'")
                    print(f"Hypothesis (norm): '{hyp_text_norm}'")
                    print(f"Match? {ref_text == hyp_text_norm}")
                    print(f"==========================\n")
                
                ref.append(ref_text)
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
    
    def test_latency(
        self,
        split: str = "test-clean",
        num_samples: int = 500,
        warmup: int = 10,
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
    ) -> Dict:
        """
        Test throughput (utterances/second) processing entire batch at once.
        """
        print(f"\n{'='*60}")
        print(f"Throughput Test: {split}")
        print(f"{'='*60}")
        samples = self.load_librispeech_subset(split, num_samples)
        reset()
        timer = Timer(device=self.device)
        with torch.inference_mode(), timer:
            for audio, sr, _ in tqdm(samples, desc="Processing"):
                _ = self.model(audio)
        total_time = timer.elapsed
        throughput = len(samples) / total_time
        memory_mb = get_gpu_memory_mb()
        results = {
            "split": split,
            "num_samples": len(samples),
            "total_time_sec": round(total_time, 2),
            "throughput_utt_per_sec": round(throughput, 2),
            "time_per_utt_ms": round((total_time / len(samples)) * 1000, 2),
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
        results_file = self.results_dir / f"baseline_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OWSM-CTC baseline model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="espnet/owsm_ctc_v4_1B",
        help="Model name",
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
        choices=["wer", "latency", "throughput", "all"],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    benchmark = BaselineBenchmark(
        model_name=args.model,
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
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()