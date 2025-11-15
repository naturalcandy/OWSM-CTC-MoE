# src/moe/patch.py
from __future__ import annotations
from typing import Iterable, Optional, Sequence

import torch.nn as nn

from moe.moe_ffn import MoEPositionwiseFFN


def get_ffn_dims(dense_ffn: nn.Module) -> tuple[int, int, float, str]:
    """Extract (d_model, hidden, dropout_p, activation_name) from ESPnet FFN."""
    d_model = dense_ffn.w_1.in_features
    hidden = dense_ffn.w_1.out_features
    dropout_p = getattr(dense_ffn, "dropout", None).p if hasattr(dense_ffn, "dropout") else 0.1
    act_name = "swish"
    return d_model, hidden, dropout_p, act_name


def inject_moe(
    encoder: nn.Module,
    n_experts: int = 8,
    top_k: int = 1,
    capacity_factor: float = 1.2,
    noisy_gate_std: float = 1.0,
    use_noisy_gating: bool = True,
    layers: Optional[Sequence[int]] = None, 
    replace_macaron: bool = True,            
    verbose: bool = True,
):
    """Replace FFNs with MoEPositionwiseFFN in an OWSM-CTC E-Branchformer encoder.

    Args:
        encoder: EBranchformerCTCEncoder (encoder.encoders is a sequence of layers)
    """
    enc_layers: Iterable[nn.Module] = encoder.encoders  
    chosen = set(layers) if layers is not None else None

    for idx, block in enumerate(enc_layers):
        if (chosen is not None) and (idx not in chosen):
            continue

        # Replace the main FFN
        if getattr(block, "feed_forward", None) is not None:
            d_model, hidden, drop_p, act = get_ffn_dims(block.feed_forward)
            moe = MoEPositionwiseFFN(
                d_model=d_model,
                hidden=hidden,
                n_experts=n_experts,
                top_k=top_k,
                dropout=drop_p,
                activation=act,
                capacity_factor=capacity_factor,
                noisy_gate_std=noisy_gate_std,
                use_noisy_gating=use_noisy_gating,
            )
            block.feed_forward = moe
            if verbose:
                print(f"[MoE] Layer {idx}: replaced feed_forward (d={d_model}, h={hidden})")

        # Replace the macaron FFN
        if replace_macaron and getattr(block, "feed_forward_macaron", None) is not None:
            d_model, hidden, drop_p, act = get_ffn_dims(block.feed_forward_macaron)
            moe2 = MoEPositionwiseFFN(
                d_model=d_model,
                hidden=hidden,
                n_experts=n_experts,
                top_k=top_k,
                dropout=drop_p,
                activation=act,
                capacity_factor=capacity_factor,
                noisy_gate_std=noisy_gate_std,
                use_noisy_gating=use_noisy_gating,
            )
            block.feed_forward_macaron = moe2
            if verbose:
                print(f"[MoE] Layer {idx}: replaced feed_forward_macaron (d={d_model}, h={hidden})")
