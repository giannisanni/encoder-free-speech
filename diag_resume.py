"""Diagnose why resume didn't apply weights: compare ckpt keys vs model keys,
do the real load_state_dict, and report missing/unexpected counts."""
import torch
from gemma_csm.model import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in

cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha = 32, 16
cfg.lora_all_linear = True
cfg.depth_loss_frame_frac = 0.25
cfg.depth_loss_weight = 0.01

print("building model ...", flush=True)
model, tok, audio_encode = load_gemma_csm_audio_in(cfg)
sd = torch.load("checkpoints/a1_speak/ckpt_6000.pt", weights_only=True, map_location="cpu")
own = model.state_dict()

ck, mk = set(sd), set(own)
print(f"ckpt keys={len(ck)}  model keys={len(mk)}")
print(f"in BOTH (will load)      = {len(ck & mk)}")
print(f"in ckpt NOT in model     = {len(ck - mk)}")
print("   e.g.", list(ck - mk)[:4])
trainable_own = {k for k in mk if "lora" in k or not k.startswith("backbone.")}
print(f"model trainable keys     = {len(trainable_own)}")
print(f"   trainable present in ckpt = {len(trainable_own & ck)}")
print("   trainable MISSING from ckpt e.g.", list(trainable_own - ck)[:4])

# shape-mismatch filter (same as train code) then real load
for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
    del sd[k]
res = model.load_state_dict(sd, strict=False)
print(f"after load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
print("   unexpected e.g.", list(res.unexpected_keys)[:4])
