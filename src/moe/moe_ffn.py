from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict

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
    """Single expert: Position-wise FFN used in ESPnet

    NOTE: We do not use this one anymore. I just keep it for reference.
    """

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


################################################################################
# Router Implementation
@dataclass
class RouterOutputs:
    topk_ids: torch.Tensor
    topk_gates: torch.Tensor
    aux_loss: torch.Tensor
    mask: torch.Tensor
    expert_counts: torch.Tensor
    router_probs: torch.Tensor


@dataclass
class MoEStats:
    """Container for MoE statistics collected during forward pass."""
    # (n_experts,) raw token counts per expert
    expert_counts: torch.Tensor
    # (n_experts,) mean routing probability
    expert_probs: torch.Tensor
    # tokens dropped due to capacity
    tokens_dropped: int
    # total tokens processed
    total_tokens: int
    # entropy of routing distribution
    router_entropy: float


class NoisyTopKRouter(nn.Module):
    """Top-k router w/ optional Gaussian noise & Switch-style aux loss.
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
        expert_dropout: float = 0.0,
        use_lang_bias: bool = False,
    ):
        super().__init__()
        assert top_k >= 1
        self.n_experts = n_experts
        self.top_k = top_k
        self.noisy_gate_std = noisy_gate_std
        self.capacity_factor = capacity_factor
        self.use_noisy_gating = use_noisy_gating
        self.router = nn.Linear(d_model, n_experts, bias=True)
        self.expert_dropout = expert_dropout
        self.use_lang_bias = use_lang_bias
        if use_lang_bias:
            # Bias
            init_bias = torch.tensor([
                [ 0.15, -0.15,  0.15, -0.15],  
                [-0.15,  0.15, -0.15,  0.15],  
            ])
            self.register_buffer('lang_bias', init_bias)

    @torch.no_grad()
    def _per_expert_capacity(self, n_tokens: int) -> int:
        # From switch transformer paper
        return int(math.ceil(self.capacity_factor * n_tokens / self.n_experts))

    def forward(self, x: torch.Tensor, training: bool, lang_families: torch.Tensor = None) -> RouterOutputs:
        # Cast to float32 for numerical stability during mixed precision training
        # (recall in section 2.4 of Switch Transformer paper)
        original_dtype = x.dtype
        if training and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        # logits is (N,E) for N tokens and E experts
        logits = self.router(x)
        if self.use_lang_bias and lang_families is not None:
            bias = self.lang_bias[lang_families]  # (N,) -> (N, E)
            logits = logits + bias
        
        # (our noisy gating)
        if training and self.expert_dropout > 0.0:
            mask = torch.rand(self.n_experts, device=x.device) > self.expert_dropout
            
            # For safety
            if mask.sum() == 0:
                mask[torch.randint(0, self.n_experts, (1,)).item()] = True
            logits = logits.masked_fill(~mask, -1e9)

        if training and self.use_noisy_gating and self.noisy_gate_std > 0.0:
            noise = torch.randn_like(logits) * self.noisy_gate_std
            logits = logits + noise

        # Get top-k expert per token (via softmax)
        probs = F.softmax(logits, dim=-1)
        topk_gates, topk_ids = torch.topk(probs, k=self.top_k, dim=-1)
        norm = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)
        renorm_gates = norm.detach() + topk_gates - topk_gates.detach()
        if training and original_dtype in (torch.float16, torch.bfloat16):
            renorm_gates = renorm_gates.to(original_dtype)

        # During inference we skip aux loss calculation and capacity enforcement
        # Not sure if this will be desired in all cases.
        if not training:
            with torch.no_grad():
                expert_counts = torch.bincount(
                    topk_ids.view(-1), minlength=self.n_experts
                ).float()
                router_probs = probs.mean(dim=0).detach()
            return RouterOutputs(
                topk_ids=topk_ids.long(),
                topk_gates=renorm_gates,
                aux_loss=torch.zeros((), device=x.device),
                mask=torch.ones_like(topk_ids, dtype=torch.bool),
                expert_counts=expert_counts,
                router_probs=router_probs,
            )
        # Compute per-expert positions for capacity enforcement
        # e.g. 1k tokens 8 experts cap factor of 1.2 means each expert
        # is capped at 150 tokens
        N, K = topk_ids.shape
        cap = self._per_expert_capacity(N)

        # Should we drop tokens or re-route??? Probs should change this..
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
        importance = probs.sum(dim=0)
        # load: how many tokens (after capacity) go to each expert
        load = torch.zeros(self.n_experts, device=device, dtype=probs.dtype)
        for e in range(self.n_experts):
            load[e] = (flat_ids == e).logical_and(keep_mask.view(-1)).sum()
        # from the switch paper, normalize by total tokens:
        # E * sum(importance*load) / N^2
        Nf = float(N) if N > 0 else 1.0
        # We essentially are penalizing imbalanced experts here.
        aux_loss = (self.n_experts * (importance / Nf) * (load / Nf)).sum()

        # For expert distribution stats
        expert_counts = load.clone().detach()
        router_probs = (probs.mean(dim=0)).detach()

        return RouterOutputs(
            topk_ids=topk_ids.long(),
            topk_gates=renorm_gates,
            aux_loss=aux_loss,
            mask=keep_mask,
            expert_counts=expert_counts,
            router_probs=router_probs,
        )


#################################################################################
# MoE Positionwise FFN with stacked weights


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
        use_expert_batching: bool = False,  # not used currently
        block_size: int = 128,  # not used currently
        expert_dropout: float = 0.0,
        use_lang_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.dropout_p = dropout
        self.activation_name = activation

        self.router = NoisyTopKRouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            noisy_gate_std=noisy_gate_std,
            capacity_factor=capacity_factor,
            use_noisy_gating=use_noisy_gating,
            expert_dropout=expert_dropout,
            use_lang_bias=use_lang_bias,
        )

        # Here we use stacked weights instead of a module list of experts
        # so all expert weights are in contiguous memory. This improve memory
        # locality and access patterns during dispatch as well as reduce
        # number of kernel launches. This is especially helpful for top-k=1
        # inference.

        # This is actually what many inference engines like SGLang do for MoE.
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.b1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        # retrieved by trainer
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_tokens_dropped: int = 0
        self.last_total_tokens: int = 0

    # TODO
    def _init_weights(self):
        """
        Follow same intialziation as nn.Linear (doesn't really matter though since
        we init from pretrained ffn.
        """
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            fan_in_w1, _ = nn.init._calculate_fan_in_and_fan_out(self.w1[i])
            bound_w1 = 1 / math.sqrt(fan_in_w1) if fan_in_w1 > 0 else 0
            nn.init.uniform_(self.b1[i], -bound_w1, bound_w1)
            fan_in_w2, _ = nn.init._calculate_fan_in_and_fan_out(self.w2[i])
            bound_w2 = 1 / math.sqrt(fan_in_w2) if fan_in_w2 > 0 else 0
            nn.init.uniform_(self.b2[i], -bound_w2, bound_w2)

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        B, T, D = x_btd.shape
        x = x_btd.reshape(B * T, D)
        lang_families = None
        if hasattr(self, 'current_lang_families') and self.current_lang_families is not None:
            lang_families = self.current_lang_families.repeat_interleave(T)
        # Router picks experts
        r = self.router(x, training=self.training, lang_families=lang_families)
        ids = r.topk_ids
        gates = r.topk_gates
        mask = r.mask

        # Store stats for trainer to consume
        self.last_expert_counts = r.expert_counts
        self.last_router_probs = r.router_probs
        self.last_tokens_dropped = int((~mask).sum().item()) if self.training else 0
        self.last_total_tokens = B * T

        # Dispatch to experts
        if self.top_k == 1:
            # Use optimized top-k=1 path (handles both training and inference)
            y = self._dispatch_stacked_topk1(
                x,
                ids.squeeze(-1),
                gates.squeeze(-1),
                mask=mask.squeeze(-1) if self.training else None,
            )
        else:
            # Use general path for top-k > 1
            y = self._dispatch_stacked_batched(x, ids, gates, mask)

        # Store aux loss for the trainer to consume
        self.last_aux_loss = r.aux_loss
        return y.reshape(B, T, D)

    def _dispatch_stacked_topk1(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized dispatch for top-k=1 using stacked weights.

        Supports both inference (no capacity) and training (with capacity mask).
        """
        N, D = x.shape

        # Handle capacity mask if provided (training with drops)
        if mask is not None:
            # mask is (N,) bool tensor
            mask_flat = mask.view(-1).bool()
            keep_idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)

            if keep_idx.numel() == 0:
                # All tokens dropped by capacity enforcement
                return torch.zeros_like(x)

            x_filtered = x[keep_idx]
            expert_ids_filtered = expert_ids[keep_idx]
            gate_weights_filtered = gate_weights[keep_idx]
        else:
            # No capacity constraints (inference)
            keep_idx = None
            x_filtered = x
            expert_ids_filtered = expert_ids
            gate_weights_filtered = gate_weights

        # Sort tokens by expert for coalesced memory access
        sorted_expert_ids, sort_indices = torch.sort(expert_ids_filtered)
        sorted_x = x_filtered[sort_indices]
        sorted_gates = gate_weights_filtered[sort_indices]

        # Find boundaries per expert
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )

        # Pre-allocate output buffer
        sorted_y = torch.zeros_like(sorted_x)

        offset = 0
        for e, cnt in zip(unique_experts.tolist(), counts.tolist()):
            start, end = offset, offset + cnt
            offset = end

            x_e = sorted_x[start:end]
            g_e = sorted_gates[start:end].unsqueeze(-1)

            # First linear: w1[e] is (hidden, D), x_e is (M, D) -> (M, hidden)
            h = F.linear(x_e, self.w1[e], self.b1[e])
            h = self.act(h)
            h = self.dropout(h)

            # Second linear: w2[e] is (D, hidden), h is (M, hidden) -> (M, D)
            out = F.linear(h, self.w2[e], self.b2[e])
            out = self.dropout(out)

            sorted_y[start:end] = out * g_e

        # Unsort and place back into original positions
        if keep_idx is not None:
            # Training with drops: scatter to original kept positions
            y = torch.zeros_like(x)
            inverse_indices = torch.argsort(sort_indices)
            y[keep_idx] = sorted_y[inverse_indices]
        else:
            # No drops: direct unsort to original order
            inverse_indices = torch.argsort(sort_indices)
            y = sorted_y[inverse_indices]

        return y

    # Dont use for now, we will just stick to topk = 1
    def _dispatch_stacked_batched(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        gates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stacked-weight dispatch for training (top-k >= 1 with capacity mask).

        1. Flatten (N, K) -> (N*K) and drop masked positions.
        2. Sort by expert for coalesced access.
        3. Process each expert with stacked weights.
        4. Scatter results back via index_add_ with gate weights.
        """
        N, D = x.shape
        N_ids, K = ids.shape
        assert N_ids == N, "ids must have shape (N, K)"

        y = x.new_zeros((N, D))

        # Flatten along (token, top-k slot)
        flat_ids = ids.contiguous().view(-1)
        flat_mask = mask.contiguous().view(-1)
        flat_gates = gates.contiguous().view(-1)

        # Positions that survive capacity mask
        active_pos = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
        if active_pos.numel() == 0:
            # Everything dropped – just return zeros
            return y

        # Recover token indices, expert ids, and gate weights for active positions
        tok_idx = active_pos // K
        expert_idx = flat_ids[active_pos].long()
        gate_w = flat_gates[active_pos]

        # Sort by expert to get contiguous segments per expert
        sorted_expert, sort_order = torch.sort(expert_idx)
        sorted_tok = tok_idx[sort_order]
        sorted_gate = gate_w[sort_order]

        # Find boundaries for each used expert
        unique_expert, counts = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )

        offset = 0
        for e, cnt in zip(unique_expert.tolist(), counts.tolist()):
            start = offset
            end = offset + cnt
            offset = end

            tok_e = sorted_tok[start:end]
            gate_e = sorted_gate[start:end]

            # Collect all tokens routed to this expert
            x_e = x[tok_e]

            # First linear using stacked weights
            h = F.linear(x_e, self.w1[e], self.b1[e])
            h = self.act(h)
            h = self.dropout(h)

            # Second linear using stacked weights
            y_e = F.linear(h, self.w2[e], self.b2[e])
            y_e = self.dropout(y_e)

            # Weighted accumulation back into y
            y.index_add_(0, tok_e, y_e * gate_e.unsqueeze(-1))

        return y


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


def collect_moe_stats(module: nn.Module) -> Dict[str, float]:
    stats = {}

    total_expert_counts = None
    total_tokens = 0
    total_dropped = 0
    n_experts = 0

    for name, m in module.named_modules():
        if isinstance(m, MoEPositionwiseFFN) and m.last_expert_counts is not None:
            n_experts = m.n_experts
            parts = name.split(".")
            layer_num = None
            for i, p in enumerate(parts):
                if p == "encoders" and i + 1 < len(parts):
                    layer_num = parts[i + 1]
                    break

            if layer_num is not None:
                if "macaron" in name:
                    prefix = f"moe_enc{layer_num}_macaron_"
                else:
                    prefix = f"moe_enc{layer_num}_ffn_"
            else:
                prefix = f"moe_{name.replace('. ', '_')}_"

            counts = m.last_expert_counts.detach().cpu()
            probs = m.last_router_probs.detach().cpu()

            total_in_layer = counts.sum().item()
            if total_in_layer > 0:
                utilization = counts / total_in_layer
                entropy = (
                    -((utilization + 1e-10) * torch.log(utilization + 1e-10))
                    .sum()
                    .item()
                )
                stats[f"{prefix}entropy"] = entropy
                max_entropy = math.log(n_experts)
                stats[f"{prefix}entropy_ratio"] = entropy / max_entropy

            if m.last_total_tokens > 0:
                stats[f"{prefix}drop_rate"] = (
                    m.last_tokens_dropped / m.last_total_tokens
                )

            mean_count = counts.float().mean()
            if mean_count > 0:
                stats[f"{prefix}load_cv"] = (counts.float().std() / mean_count).item()

            if total_expert_counts is None:
                total_expert_counts = counts.clone()
            else:
                total_expert_counts += counts
            total_tokens += m.last_total_tokens
            total_dropped += m.last_tokens_dropped

    # Global stats
    if total_expert_counts is not None and total_tokens > 0:
        stats["moe_total_drop_rate"] = total_dropped / total_tokens
        global_util = total_expert_counts / total_expert_counts.sum()
        global_entropy = (
            -((global_util + 1e-10) * torch.log(global_util + 1e-10)).sum().item()
        )
        stats["moe_global_entropy"] = global_entropy
        stats["moe_global_entropy_ratio"] = global_entropy / math.log(n_experts)

        for i in range(n_experts):
            stats[f"moe_expert{i}_total_frac"] = (
                total_expert_counts[i] / total_expert_counts.sum()
            ).item()

    return stats
