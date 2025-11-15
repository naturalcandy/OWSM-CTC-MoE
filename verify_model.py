from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

print("Loading OWSM-CTC v4 (1B)...")
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v4_1B",
    device="cuda",
    use_flash_attn=False,
    lang_sym="<eng>",
    task_sym="<asr>",
)

model = s2t.s2t_model
total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params/1e9:.2f}B parameters")