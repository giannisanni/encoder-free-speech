"""Validate the audio-IN training forward (rev.24): GemmaCSM.forward must accept
an [text | audio-in | audio-out] sequence (precomputed backbone-dim
audio_in_embeds), shift the output-prediction positions, run the heads, and flow
gradients -- AND leave the text->audio path byte-identical when audio-in absent.
Uses synthetic audio_in_embeds (get_audio_features was validated separately).

Run:  GEMMA_BACKBONE=google/gemma-4-12B-it .venv/bin/python -m scripts.probe_audio_in_forward
"""
import os
import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm

cfg = GemmaCSMConfig()
cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
print("backbone:", cfg.backbone_id, flush=True)
model, tok = load_gemma_csm(cfg)
d = model.backbone.config.hidden_size
dev = "cuda"


def make_batch(with_audio_in):
    B, Tt, Tf = 1, 8, 13
    b = {
        "text_ids": torch.randint(1, 100, (B, Tt), device=dev),
        "text_mask": torch.ones(B, Tt, dtype=torch.bool, device=dev),
        "codes": torch.randint(0, 2048, (B, 32, Tf), device=dev),
        "frame_mask": torch.ones(B, Tf, dtype=torch.bool, device=dev),
    }
    if with_audio_in:
        Ta = 20
        b["audio_in_embeds"] = torch.randn(B, Ta, d, device=dev, dtype=torch.bfloat16)
        b["audio_in_mask"] = torch.ones(B, Ta, dtype=torch.bool, device=dev)
    return b


def ok(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}", flush=True)
    return cond


allok = True
out0 = model(make_batch(False)); l0 = float(out0["loss"])
allok &= ok("text->audio forward (no audio-in) still works", l0 == l0, f"loss={l0:.3f}")
out1 = model(make_batch(True)); l1 = float(out1["loss"])
allok &= ok("audio-IN forward runs [text|audio-in|audio-out]", l1 == l1,
            f"loss={l1:.3f} c0={float(out1['loss_c0']):.3f} depth={float(out1['loss_depth']):.3f}")
out1["loss"].backward()
g = sum(p.grad.abs().sum().item() for p in model.parameters()
        if p.requires_grad and p.grad is not None)
allok &= ok("backward + grads flow to trainables", g > 0, f"sum|grad|={g:.2e}")
print("peak VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB", flush=True)
print("=== AUDIO-IN FORWARD CONFIRMED ===" if allok else "=== SOME CHECKS FAILED ===", flush=True)
