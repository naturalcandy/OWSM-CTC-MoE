from espnet2.s2t.espnet_ctc_model import ESPnetS2TCTCModel
from espnet2.torch_utils.device_funcs import force_gatherable
from moe.moe_ffn import collect_moe_aux_loss, collect_moe_stats, MoEPositionwiseFFN
import torch


# Language family mappings
LANG_TO_FAMILY = {
    "spa": 0, "por": 0, "ita": 0, "fra": 0,  # Romance = 0
    "pol": 1, "ces": 1, "slk": 1, "hrv": 1,  # Slavic = 1
}

class MoEESPnetS2TCTCModel(ESPnetS2TCTCModel):
    """OWSM-CTC model with MoE auxiliary loss, stats tracking, and optional language-biased routing."""

    def __init__(self, *args, moe_aux_weight: float = 0.01, use_lang_bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_aux_weight = moe_aux_weight
        self.use_lang_bias = use_lang_bias
        
        # Token ID -> family ID mapping (populated by set_language_token_mapping)
        self._lang_token_to_family = {}  # token_id -> 0 (Romance) or 1 (Slavic)
    
    def set_language_token_mapping(self, tokenizer, converter):
        """
        Build mapping from language token IDs to family indices.
        Call this once after model is loaded with the tokenizer.
        """
        for iso3, family_id in LANG_TO_FAMILY.items():
            lang_token = f"<{iso3}>"
            try:
                token_ids = converter.tokens2ids(tokenizer.text2tokens(lang_token))
                if len(token_ids) >= 2:
                    self._lang_token_to_family[token_ids[1]] = family_id
                elif token_ids:
                    self._lang_token_to_family[token_ids[0]] = family_id
            except Exception:
                pass
        
        if self._lang_token_to_family:
            print(f"[MoE] Language token mapping: {len(self._lang_token_to_family)} tokens mapped")
            for tok_id, fam_id in self._lang_token_to_family.items():
                print(f"      Token ID {tok_id} -> Family {fam_id} ({'Romance' if fam_id == 0 else 'Slavic'})")
    
    def _extract_lang_families(self, prefix: torch.Tensor) -> torch.Tensor:
        """Extract language family for each sample in batch."""
        B = prefix.shape[0]
        device = prefix.device
        families = torch.zeros(B, dtype=torch.long, device=device)
        
        if self._lang_token_to_family:
            for i in range(B):
                for pos in range(min(3, prefix.shape[1])):
                    token_id = prefix[i, pos].item()
                    if token_id in self._lang_token_to_family:
                        families[i] = self._lang_token_to_family[token_id]
                        break
        
        return families
    
    def _set_lang_family_on_moe_modules(self, lang_family):
        """Set the current language family on all MoE modules."""
        for module in self.encoder.modules():
            if isinstance(module, MoEPositionwiseFFN):
                module.current_lang_family = lang_family

    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        text_prev,
        text_prev_lengths,
        text_ctc,
        text_ctc_lengths,
        prefix,
        prefix_lengths,
        **kwargs,
    ):
        lang_families = self._extract_lang_families(prefix)
        for m in self.encoder.modules():
            if isinstance(m, MoEPositionwiseFFN):
                m.current_lang_families = lang_families
        
        # Call parent forward to get base loss
        loss, stats, weight = super().forward(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            text_prev=text_prev,
            text_prev_lengths=text_prev_lengths,
            text_ctc=text_ctc,
            text_ctc_lengths=text_ctc_lengths,
            prefix=prefix,
            prefix_lengths=prefix_lengths,
            **kwargs,
        )

        # Collect MoE auxiliary loss
        moe_aux_loss = collect_moe_aux_loss(self.encoder)

        # Add weighted aux loss to total loss
        loss = loss + self.moe_aux_weight * moe_aux_loss

        moe_stats = collect_moe_stats(self.encoder)

        # Update stats dict - ESPnet trainer will log these to TensorBoard
        stats["loss_moe_aux"] = moe_aux_loss.detach()
        stats["loss"] = loss.detach()

        # Add all MoE stats (entropy, drop rates, load balance, etc.)
        for key, value in moe_stats.items():
            stats[key] = value

        # Re-apply force_gatherable for DataParallel compatibility
        batch_size = speech.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight
