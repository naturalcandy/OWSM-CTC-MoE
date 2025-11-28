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

    args = parser.parse_args()
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OWSM-CTC Fine-tuning on LibriSpeech-100h")
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
    
    def to_speech(sample):
        waveform, sample_rate, transcript, *_ = sample
        audio = waveform[0].numpy().astype(np.float32)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sample_rate, 16000
            ).numpy().astype(np.float32)
        return audio
    
    def get_transcript(sample):
        """
        Normalize LibriSpeech transcript to match OWSM-CTC output format.
        
        Model outputs: "He hoped there would be stew for dinner, ..."
        So we train on: "he hoped there would be stew for dinner, ..."
        """
        _, _, transcript, *_ = sample
        transcript = transcript.lower()
        if not transcript.endswith('.'):
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
    
    print("\n[2/4] Loading datasets...")
    train_raw = torchaudio.datasets.LIBRISPEECH(
        str(args.data_dir), url="train-clean-100", download=False
    )
    valid_raw = torchaudio.datasets.LIBRISPEECH(
        str(args.data_dir), url="dev-clean", download=False
    )
    
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
    print(f"Monitor: tensorboard --logdir {args.exp_dir}/tensorboard")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()