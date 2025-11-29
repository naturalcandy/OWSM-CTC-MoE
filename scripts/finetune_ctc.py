"""
Fine-tune OWSM-CTC v4 1B on LibriSpeech train-clean-100
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import espnetez as ez
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

from moe.moe_ffn import MoEPositionwiseFFN
from datasets import load_dataset
################################################
# Fleurs setup

FLEURS_LANGUAGES = {
    # Training languages
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

TRAIN_LANGUAGES = [k for k, v in FLEURS_LANGUAGES.items() if v[2] == "train"]


class MultilingualFLEURSDataset:
    """
    Combined dataset for multiple FLEURS languages.
    """
    
    def __init__(self, language_codes: list, split: str, cache_dir: str):
        self.samples = []
        # index -> iso3 code
        self.lang_map = {}  
        idx = 0
        for fleurs_code in language_codes:
            iso3, family, role = FLEURS_LANGUAGES[fleurs_code]
            print(f"  Loading {fleurs_code} ({iso3}) - {split}...")
            ds = load_dataset(
                "google/fleurs",
                fleurs_code,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            for sample in ds:
                self.samples.append({
                    "audio": sample["audio"],
                    "transcription": sample["transcription"],
                    "iso3": iso3,
                    "fleurs_code": fleurs_code,
                })
                self.lang_map[idx] = iso3
                idx += 1
            print(f"    -> {len(ds)} samples")
        print(f"  Total: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


################################################


# Sourced from EspnetEZ finetuning tutorial:
# https://espnet.github.io/espnet/notebook/ESPnetEZ/ASR/ASR_finetune_owsm.html

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path("exp/finetune_ls100h"))
    parser.add_argument("--stats_dir", type=Path, default=Path("exp/stats_finetune"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/librispeech"))
    parser.add_argument("--model_name", type=str, default="espnet/owsm_ctc_v4_1B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--config", type=Path, default=Path("config/finetune_ctc.yaml"))
    parser.add_argument("--inject_moe", action="store_true")
    parser.add_argument("--dataset", type=str, choices=["librispeech", "fleurs"], 
                        default="librispeech")
    parser.add_argument("--fleurs_dir", type=Path, default=Path("data/fleurs"))
    parser.add_argument("--fleurs_langs", type=str, nargs="+", default=None)

    args = parser.parse_args()
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    if args.dataset == "librispeech":
        print("OWSM-CTC Fine-tuning on LibriSpeech-100h")
    else:
        print("OWSM-CTC Fine-tuning on Multilingual FLEURS")
    print("="*80)
    print("\n[1/4] Loading pretrained model...")
    s2t = Speech2TextGreedySearch.from_pretrained(
        args.model_name,
        device=args.device,
        lang_sym=f"<{args.lang}>",
        task_sym="<asr>",
        use_flash_attn=True,
    )
    torch.save(s2t.s2t_model.state_dict(), args.exp_dir / "original.pth")
    print(f"Saved original weights")
    
    pretrain_config = vars(s2t.s2t_train_args)
    tokenizer = s2t.tokenizer
    converter = s2t.converter
    pretrained_model = s2t
    
    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    print("\n[2/4] Loading datasets...")

    if args.dataset == "librispeech":
        def to_speech(sample):
            waveform, sample_rate, transcript, *_ = sample
            audio = waveform[0].numpy().astype(np.float32)
            if sample_rate != 16000:
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sample_rate, 16000
                ).numpy().astype(np.float32)
            return audio
        
        def get_transcript(sample):
            _, _, transcript, *_ = sample
            transcript = transcript.lower()
            if not transcript.endswith('. '):
                transcript = transcript + '.'
            return transcript
        
        data_info = {
            "speech": lambda d: to_speech(d),
            # Model output format is: "<eng><asr> Text..."
            "text": lambda d: tokenize(f"<{args.lang}><asr> {get_transcript(d)}"),
            "text_prev": lambda d: tokenize("<na>"),
            "text_ctc": lambda d: tokenize(get_transcript(d)),
            "prefix": lambda d: tokenize(f"<{args.lang}><asr>"),
        }
        
        train_raw = torchaudio.datasets.LIBRISPEECH(
            str(args.data_dir), url="train-clean-100", download=False
        )
        valid_raw = torchaudio.datasets.LIBRISPEECH(
            str(args.data_dir), url="dev-clean", download=False
        )
        
        train_dataset = ez.dataset.ESPnetEZDataset(train_raw, data_info=data_info)
        valid_dataset = ez.dataset.ESPnetEZDataset(valid_raw, data_info=data_info)
    
    elif args.dataset == "fleurs":
        train_langs = args.fleurs_langs if args.fleurs_langs else TRAIN_LANGUAGES
        print(f"Training languages: {train_langs}")
        for lang in train_langs:
            iso3, family, role = FLEURS_LANGUAGES[lang]
            print(f"  {lang} -> <{iso3}> ({family})")
        
        print("\nLoading training set:")
        train_raw = MultilingualFLEURSDataset(
            language_codes=train_langs,
            split="train",
            cache_dir=str(args.fleurs_dir),
        )
        
        print("\nLoading validation set:")
        valid_raw = MultilingualFLEURSDataset(
            language_codes=train_langs,
            split="validation",
            cache_dir=str(args.fleurs_dir),
        )
        
        # FLEURS data_info (language is per-sample)
        def fleurs_to_speech(sample):
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            return audio
        
        def fleurs_get_transcript(sample):
            return sample["transcription"]
        
        def fleurs_get_iso3(sample):
            return sample["iso3"]
        
        data_info = {
            "speech": lambda d: fleurs_to_speech(d),
            # Per-sample language token
            "text": lambda d: tokenize(f"<{fleurs_get_iso3(d)}><asr> {fleurs_get_transcript(d)}"),
            "text_prev": lambda d: tokenize("<na>"),
            "text_ctc": lambda d: tokenize(fleurs_get_transcript(d)),
            "prefix": lambda d: tokenize(f"<{fleurs_get_iso3(d)}><asr>"),
        }
        train_dataset = ez.dataset.ESPnetEZDataset(train_raw, data_info=data_info)
        valid_dataset = ez.dataset.ESPnetEZDataset(valid_raw, data_info=data_info)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Valid: {len(valid_dataset)} samples")
    
    print("\n[3/4] Configuring training...")
    finetune_config = ez.config.update_finetune_config(
        "s2t",
        pretrain_config,
        str(args.config),
    )
    print(f"Epochs: {finetune_config.get('max_epoch')}")
    print(f"Batch size: {finetune_config.get('batch_size')}")

    use_moe = args.inject_moe
    
    def build_model_fn(args):
        model = pretrained_model.s2t_model
        if use_moe and not any(isinstance(m, MoEPositionwiseFFN) 
                           for m in model.encoder.modules()):
            from moe.patch import inject_moe
            from moe.moe_ctc import MoEESPnetS2TCTCModel
            print("\nInjecting MoE layers into last 9 encoder layers...")
            num_layers = len(model.encoder.encoders)
            target_layers = list(range(num_layers - 9, num_layers))
            inject_moe(
                model.encoder,
                n_experts=8,
                top_k=1,
                capacity_factor=2.5, 
                noisy_gate_std=0.25,
                use_noisy_gating=True,
                layers=target_layers,
                replace_macaron=True,
                init_from_pretrained=True,
                init_noise_std=0.01,
                verbose=True,
            )
            print("MoE injection complete.\n")
            moe_count = sum(1 for m in model.encoder.modules() 
                            if isinstance(m, MoEPositionwiseFFN))
            print(f"Verified: {moe_count} MoE layers injected")
            assert moe_count == len(target_layers) * 2, "MoE injection count mismatch!"
            # we also need to incorpoate auxiliary loss for MoE during training
            model.__class__ = MoEESPnetS2TCTCModel
            model.moe_aux_weight = 0.01

        model.train()
        print(f"Trainable parameters: {count_trainable(model):,}")
        return model
    
    print("\n[4/4] Initializing trainer...")
    trainer = ez.Trainer(
        task='s2t',
        train_config=finetune_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=str(args.exp_dir),
        stats_dir=str(args.stats_dir),
        ngpu=1,
    )
    
    print("\nCollecting statistics...")
    trainer.collect_stats()
    print("Stats complete\n")
    
    print("="*80)
    print("TRAINING START")
    if args.dataset == "fleurs":
        train_langs = args.fleurs_langs if args.fleurs_langs else TRAIN_LANGUAGES
        print(f"Languages: {train_langs}")
    print(f"Monitor: tensorboard --logdir {args.exp_dir}/tensorboard")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()