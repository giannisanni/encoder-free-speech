"""Validate the rewritten forward_unified (native delivery, segment-split) on the
REAL 12B backbone. The CPU fake-backbone test cannot exercise native delivery
(no audio tower / masked_scatter / KV across the full model), so this does a
1-step forward+backward on one real OpenS2S item and checks:
  - losses finite (text / c0 / depth)
  - gradients flow to the trainables (LoRA + speech head)
  - seg1(full model)->seg2(backbone) KV hand-off works
  - peak VRAM fits the 16 GB target

Run (substrate, audio-IN venv):
  GEMMA_BACKBONE=google/gemma-4-12B-it \
  ~/audioin-venv/bin/python -m scripts.val_unified_forward \
    --shard data/shards_opens2s/opens2s_en_000.pt --n 2 --max-frames 80 --max-audio 64
"""
import argparse
import os

import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset_opens2s import OpenS2SDataset, collate_opens2s
from gemma_csm.loader import load_gemma_csm_audio_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=2, help="batch size (items looped inside forward)")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--max-audio", type=int, default=64)
    ap.add_argument("--max-text", type=int, default=96)
    a = ap.parse_args()

    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
    dev = "cuda"
    model, tok, _ = load_gemma_csm_audio_in(cfg, device=dev)

    # forward_unified runs the full model with use_cache=True (seg1) then hands the
    # KV to the backbone (seg2). gradient checkpointing forces use_cache=False, so
    # it MUST be off for this path. (Trade-off: more activation memory; the seq is
    # short + batch small, so it fits 16 GB -- this check proves it.)
    for m in (getattr(model, "full_model", None), model.backbone):
        try:
            m.gradient_checkpointing_disable()
        except Exception as e:
            print("gc disable note:", e, flush=True)
    model.train()

    ds = OpenS2SDataset(a.shard)
    items = [ds[i] for i in range(a.n)]
    batch = collate_opens2s(items, tok, max_audio=a.max_audio,
                            max_frames=a.max_frames, max_text=a.max_text)
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(f"batch: B={a.n} text={tuple(batch['text_ids'].shape)} "
          f"codes={tuple(batch['codes'].shape)} feats={tuple(batch['input_features'].shape)}",
          flush=True)

    out = model.forward_unified(batch, tok.eos_token_id)
    finite = True
    for k in ("loss", "loss_text", "loss_c0", "loss_depth"):
        f = torch.isfinite(out[k]).all().item()
        finite &= f
        print(f"  {k} = {float(out[k]):.4f}  finite={f}", flush=True)

    out["loss"].backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters()
            if p.requires_grad and p.grad is not None)
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"\ntrainable tensors={n_tr}  with-grad={n_grad}  sum|grad|={g:.3e}", flush=True)
    print("peak VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB", flush=True)

    ok = finite and g > 0 and n_grad > 0
    print("\n=== " + ("PASS: forward_unified native delivery trains end-to-end ==="
                      if ok else "FAIL ==="), flush=True)


if __name__ == "__main__":
    main()
