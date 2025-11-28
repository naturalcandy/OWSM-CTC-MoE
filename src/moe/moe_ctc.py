from espnet2.s2t.espnet_ctc_model import ESPnetS2TCTCModel
from espnet2.torch_utils.device_funcs import force_gatherable
from moe.moe_ffn import collect_moe_aux_loss, collect_moe_stats


class MoEESPnetS2TCTCModel(ESPnetS2TCTCModel):
    """OWSM-CTC model with MoE auxiliary loss and stats tracking."""

    def __init__(self, *args, moe_aux_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_aux_weight = moe_aux_weight

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

        # === ADD: Collect and log MoE stats ===
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
