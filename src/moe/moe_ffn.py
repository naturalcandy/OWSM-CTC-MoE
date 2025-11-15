# src/moe/moe_ffn.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from espnet.nets.pytorch_backend.nets_utils import get_activation
    _HAS_ESPNET_ACT = True
except Exception:
    _HAS_ESPNET_ACT = False


def _make_activation(name: str) -> nn.Module:
    if _HAS_ESPNET_ACT:
        return get_activation(name)
    raise ValueError(f"Unsupported activation: {name}")


class ExpertFFN(nn.Module):
    """Single expert: Position-wise FFN used in ESPnet"""
    def __init__(
        self,
        d_model: int,
        hidden: int,
        dropout: float = 0.1,
        activation: str = "swish",
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) where N is #tokens routed to this expert
        # time/frame agnostic
        if x.numel() == 0:
            return x  # nothing to do
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


@dataclass
class RouterOutputs:
    topk_ids: torch.Tensor      
    topk_gates: torch.Tensor  
    aux_loss: torch.Tensor      
    mask: torch.Tensor          


class NoisyTopKRouter(nn.Module):
    """Top-k router with optional Gaussian noise & Switch-style aux loss.
    - Aux loss encourages well-balanced 'importance' and 'load' across experts.
    - Capacity factor enforces per-expert token budgets, dropping overflow.
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 1,
        noisy_gate_std: float = 1.0,
        capacity_factor: float = 1.2,
        use_noisy_gating: bool = True,
    ):
        super().__init__()
        assert top_k >= 1
        self.n_experts = n_experts
        self.top_k = top_k
        self.noisy_gate_std = noisy_gate_std
        self.capacity_factor = capacity_factor
        self.use_noisy_gating = use_noisy_gating
        self.router = nn.Linear(d_model, n_experts, bias=True)

    @torch.no_grad()
    def _per_expert_capacity(self, n_tokens: int) -> int:
        # From switch transformer paper
        return int(math.ceil(self.capacity_factor * n_tokens / self.n_experts))

    def forward(self, x: torch.Tensor, training: bool) -> RouterOutputs:
        # logits is (N,E) for N tokens and E experts
        logits = self.router(x)

        # (our noisy gating)
        if training and self.use_noisy_gating and self.noisy_gate_std > 0.0:
            noise = torch.randn_like(logits) * self.noisy_gate_std
            logits = logits + noise

        # Get top-k expert per token (via softmax)
        probs = F.softmax(logits, dim=-1) 
        topk_gates, topk_ids = torch.topk(probs, k=self.top_k, dim=-1)  # (N, K)

        # Compute per-expert positions for capacity enforcement
        # e.g. 1k tokens 8 experts cap factor of 1.2 means each expert
        # is capped at 150 tokens
        N, K = topk_ids.shape
        cap = self._per_expert_capacity(N)

        # Should we drop tokens or re-route??? Probs should change this..
        # Build a mask that keeps only the top 'cap' tokens per expert (by gate)
        # Note: we sort tokens per expert by gate weight descending and keep first 'cap'.
        device = x.device
        keep_mask = torch.zeros_like(topk_ids, dtype=torch.bool, device=device)
        # Flatten (N*K) indices to iterate per expert efficiently
        flat_ids = topk_ids.view(-1)
        flat_gates = topk_gates.view(-1)
        # For each expert gather its tokens
        for e in range(self.n_experts):
            # positions in flat arrays where expert e was selected
            idx = torch.nonzero(flat_ids == e, as_tuple=False).squeeze(-1)  
            if idx.numel() == 0:
                continue
            # Sort by gate strength
            idx_sorted = idx[torch.argsort(flat_gates[idx], descending=True)]
            idx_keep = idx_sorted[:cap]
            keep_mask.view(-1)[idx_keep] = True

        # follow wih switch‑style load‑balancing auxiliary loss
        # importance = how much probability mass the router assigns to each expert
        importance = probs.sum(dim=0)    # (E,)
        # load: how many tokens (after capacity) go to each expert
        load = torch.zeros(self.n_experts, device=device, dtype=probs.dtype)
        for e in range(self.n_experts):
            load[e] = (flat_ids == e).logical_and(keep_mask.view(-1)).sum()
        # frm the switch paper, normalize by total tokens:
        #    E * sum(importance*load) / N^2
        Nf = float(N) if N > 0 else 1.0
        # We essentialyl are penalizing imbalanced experts here.
        aux_loss = (self.n_experts * (importance / Nf) * (load / Nf)).sum()

        return RouterOutputs(
            topk_ids=topk_ids.long(),
            topk_gates=topk_gates,
            aux_loss=aux_loss,
            mask=keep_mask,
        )


class MoEPositionwiseFFN(nn.Module):
    """
    Drop-in replacement for ESPnet PositionwiseFeedForward.
    Inputs:  x (B, T, D)
    Returns: y (B, T, D)
    It also sets self.last_aux_loss (tensor with grad) each forward.
    """
    def __init__(
        self,
        d_model: int,
        hidden: int,
        n_experts: int = 8,
        top_k: int = 1,
        dropout: float = 0.1,
        activation: str = "swish",
        capacity_factor: float = 1.2,
        noisy_gate_std: float = 1.0,
        use_noisy_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = NoisyTopKRouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            noisy_gate_std=noisy_gate_std,
            capacity_factor=capacity_factor,
            use_noisy_gating=use_noisy_gating,
        )
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, hidden, dropout=dropout, activation=activation)
             for _ in range(n_experts)]
        )

        self.last_aux_loss: Optional[torch.Tensor] = None  # retrieved by trainer

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        B, T, D = x_btd.shape
        x = x_btd.reshape(B * T, D) 
        N = x.shape[0]

        # Router picks experts
        r = self.router(x, training=self.training)
        ids = r.topk_ids         
        gates = r.topk_gates     
        mask = r.mask            

        # Collect per-expert token indices
        y = x.new_zeros((N, D))
        for e in range(self.n_experts):
            # tokens routed (and kept) to expert e across all K
            sel = ((ids == e) & mask)  
            if not sel.any():
                continue
            # For each occurrence, we need the gating weight
            # Flatten N*K -> positions for this expert
            pos = torch.nonzero(sel, as_tuple=False)
            tok_idx = pos[:, 0]     
            k_slot = pos[:, 1]       
            gate_w = gates[tok_idx, k_slot] 

            x_e = x[tok_idx] 
            y_e = self.experts[e](x_e)


            y.index_add_(0, tok_idx, y_e * gate_w.unsqueeze(-1))

        # Store aux loss for the trainer to consume
        self.last_aux_loss = r.aux_loss

        return y.reshape(B, T, D)


# Collects aux loss from all MoE FFN module.
def collect_moe_aux_loss(module: nn.Module) -> torch.Tensor:
    """Sum all MoE aux losses registered on MoEPositionwiseFFN modules."""
    total = None
    for m in module.modules():
        if isinstance(m, MoEPositionwiseFFN) and (m.last_aux_loss is not None):
            total = m.last_aux_loss if total is None else (total + m.last_aux_loss)
    if total is None:
        # Return a zero tensor on correct device to keep autograd happy
        total = torch.zeros((), device=next(module.parameters()).device)
    return total
