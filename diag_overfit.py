"""Test the overfitting hypothesis: with ckpt_6000 loaded, measure mean c0 on
OLD shards (000/001, trained on) vs NEW shards (002+, unseen). If old<<new,
the c0~1.7 we saw was memorization and the grown data is doing its job."""
import glob
import torch
from gemma_csm.model import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.dataset_opens2s import OpenS2SDataset, collate_opens2s
from gemma_csm.train_audio_in import evaluate

cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha = 32, 16
cfg.lora_all_linear = True
cfg.depth_loss_frame_frac = 0.25
cfg.depth_loss_weight = 0.01

print("building + loading ckpt_6000 ...", flush=True)
model, tok, audio_encode = load_gemma_csm_audio_in(cfg)
sd = torch.load("checkpoints/a1_speak_safe/ckpt_6000.pt", weights_only=True, map_location="cpu")
for k in [k for k, v in sd.items() if k in model.state_dict() and model.state_dict()[k].shape != v.shape]:
    del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()

def shard_items(pattern, n=40):
    items = []
    for f in sorted(glob.glob(pattern)):
        items += OpenS2SDataset([f]).items
    return items[:n]

old = shard_items("data/shards_opens2s/opens2s_en_00[01].pt")
new = shard_items("data/shards_opens2s/opens2s_en_00[2-9].pt")
print(f"old(trained) items={len(old)}  new(unseen) items={len(new)}", flush=True)

for name, items in [("OLD/trained", old), ("NEW/unseen", new)]:
    c0, d = evaluate(model, items, audio_encode, tok, 64, 90,
                     no_audio_in=True, max_text=56)
    print(f"{name:14s} c0={c0:.3f} depth={d:.3f}", flush=True)
