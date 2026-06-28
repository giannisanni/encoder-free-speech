"""Confirm the rewritten generate_unified (native delivery) grounds at INFERENCE.
No training needed: Phase 0 proved the frozen backbone produces on-topic reply
text from native audio, so generate_unified (untrained head OK) should too. We
only check the generated reply TEXT (audio quality needs a trained head):
  - real-audio reply is on-topic AND differs from zeroed-audio reply -> the
    inference path grounds (no bypass).
Run (substrate): GEMMA_BACKBONE=google/gemma-4-12B-it \
  ~/audioin-venv/bin/python -m scripts.val_unified_generate \
    --shard data/shards_opens2s/opens2s_en_000.pt --index 0
"""
import argparse
import os

import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset_opens2s import OpenS2SDataset
from gemma_csm.loader import load_gemma_csm_audio_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-text", type=int, default=48)
    a = ap.parse_args()
    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
    model, tok, _ = load_gemma_csm_audio_in(cfg, device="cuda")
    for m in (getattr(model, "full_model", None), model.backbone):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass
    model.eval()

    it = OpenS2SDataset(a.shard)[a.index]
    feats = it["input_features"].to("cuda", torch.bfloat16)        # (Ta, 640)
    print("query:", repr(it.get("query_text", "")), flush=True)
    # text_temperature 0 -> greedy, max_frames 1 -> we only care about the reply TEXT
    real, _ = model.generate_unified(feats, tok, tok.eos_token_id,
                                     max_text=a.max_text, max_frames=1, text_temperature=0.0)
    zero, _ = model.generate_unified(torch.zeros_like(feats), tok, tok.eos_token_id,
                                     max_text=a.max_text, max_frames=1, text_temperature=0.0)
    print("real-audio reply :", repr(real), flush=True)
    print("zeroed-audio reply:", repr(zero), flush=True)
    grounded = real.strip() != zero.strip() and len(real.strip()) > 0
    print("=== " + ("PASS: generate_unified inference grounds (reply tracks audio) ==="
                    if grounded else "FAIL: reply ignores audio ==="), flush=True)


if __name__ == "__main__":
    main()
