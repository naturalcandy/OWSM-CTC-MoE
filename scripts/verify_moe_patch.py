# scripts/verify_moe_patch.py
import sys
from pathlib import Path
import torch
import soundfile as sf
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moe import inject_moe

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s2t = Speech2TextGreedySearch.from_pretrained(
        "espnet/owsm_ctc_v4_1B",
        device=device,
        lang_sym="<eng>",
        task_sym="<asr>",
        use_flash_attn=False,
    )
    asr_model = s2t.s2t_model
    encoder = asr_model.encoder

    # Inject MoE into all blocks
    inject_moe(
        encoder,
        n_experts=8,
        top_k=1,
        capacity_factor=1.25,
        noisy_gate_std=1.0,
        use_noisy_gating=True,
        layers=None,           # all layers
        replace_macaron=True,  # replace both FFNs per block
        verbose=True,
    )

    # print model summary
    print(s2t.s2t_model.encoder)
    total_params = sum(p.numel() for p in asr_model.parameters())
    print(f"Model loaded: {total_params/1e9:.2f}B parameters")

if __name__ == "__main__":
    main()
