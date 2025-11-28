# src/moe/patch.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from moe.moe_ffn import MoEPositionwiseFFN


def _init_experts_from_ffn(
    moe: MoEPositionwiseFFN,
    pretrained_ffn: nn.Module,
    noise_std: float = 0.0,
    verbose: bool = False,
):
    """Initialize all MoE stacked expert weights from pretrained FFN weights."""
    if verbose:
        noise_msg = f" + {noise_std:.3f} noise" if noise_std > 0 else ""
        print(
            f"[MoE Init] Copying pretrained FFN weights to {moe.n_experts} experts{noise_msg}"
        )

    # ESPnet FFN has w_1 and w_2
    # w_1: Linear(d_model, hidden) -> weight shape (hidden, d_model)
    # w_2: Linear(hidden, d_model) -> weight shape (d_model, hidden)
    with torch.no_grad():
        for expert_idx in range(moe.n_experts):
            # Copy w1: pretrained w_1.weight is (hidden, d_model)
            # Our moe.w1[expert_idx] is also (hidden, d_model)
            moe.w1.data[expert_idx].copy_(pretrained_ffn.w_1.weight.data)
            moe.b1.data[expert_idx].copy_(pretrained_ffn.w_1.bias.data)

            # Copy w2: pretrained w_2.weight is (d_model, hidden)
            # Our moe.w2[expert_idx] is also (d_model, hidden)
            moe.w2.data[expert_idx].copy_(pretrained_ffn.w_2.weight.data)
            moe.b2.data[expert_idx].copy_(pretrained_ffn.w_2.bias.data)

            # Add noise
            if noise_std > 0:
                moe.w1.data[expert_idx].add_(
                    torch.randn_like(moe.w1[expert_idx]) * noise_std
                )
                moe.b1.data[expert_idx].add_(
                    torch.randn_like(moe.b1[expert_idx]) * noise_std
                )
                moe.w2.data[expert_idx].add_(
                    torch.randn_like(moe.w2[expert_idx]) * noise_std
                )
                moe.b2.data[expert_idx].add_(
                    torch.randn_like(moe.b2[expert_idx]) * noise_std
                )


def get_ffn_dims(dense_ffn: nn.Module) -> tuple[int, int, float, str]:
    """Extract (d_model, hidden, dropout_p, activation_name) from ESPnet FFN."""
    d_model = dense_ffn.w_1.in_features
    hidden = dense_ffn.w_1.out_features
    dropout_p = (
        getattr(dense_ffn, "dropout", None).p if hasattr(dense_ffn, "dropout") else 0.1
    )
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
    init_from_pretrained: bool = True,
    init_noise_std: float = 0.0,
    verbose: bool = True,
    use_expert_batching: bool = False,  # not used
):
    """Replace FFNs with MoEPositionwiseFFN in an OWSM-CTC E-Branchformer encoder."""
    enc_layers: Iterable[nn.Module] = encoder.encoders
    chosen = set(layers) if layers is not None else None

    for idx, block in enumerate(enc_layers):
        if (chosen is not None) and (idx not in chosen):
            continue

        # Replace the main FFN
        if getattr(block, "feed_forward", None) is not None:
            old_ffn = block.feed_forward
            d_model, hidden, drop_p, act = get_ffn_dims(old_ffn)

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

            # Initialize experts from pretrained FFN weights
            if init_from_pretrained:
                _init_experts_from_ffn(
                    moe, old_ffn, init_noise_std, verbose and idx == 0
                )

            block.feed_forward = moe

            if verbose:
                init_msg = (
                    " (init from pretrained)"
                    if init_from_pretrained
                    else " (random init)"
                )
                print(
                    f"[MoE] Layer {idx}: replaced feed_forward (d={d_model}, h={hidden}){init_msg}"
                )

        # Replace the macaron FFN
        if replace_macaron and getattr(block, "feed_forward_macaron", None) is not None:
            old_macaron = block.feed_forward_macaron
            d_model, hidden, drop_p, act = get_ffn_dims(old_macaron)

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
            # Initialize experts from pretrained FFN weights
            if init_from_pretrained:
                _init_experts_from_ffn(
                    moe2, old_macaron, init_noise_std, verbose and idx == 0
                )

            block.feed_forward_macaron = moe2
            if verbose:
                init_msg = (
                    " (init from pretrained)"
                    if init_from_pretrained
                    else " (random init)"
                )
                print(
                    f"[MoE] Layer {idx}: replaced feed_forward_macaron (d={d_model}, h={hidden}){init_msg}"
                )
