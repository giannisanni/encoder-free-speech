"""Regression test for train/inference alignment.

If training conditions the depth decoder on a hidden state that already
contains the target frame (leakage), a model that has perfectly memorized
one clip will still generate wrong fine codebooks at inference. Overfit a
tiny model on a single clip and require generation to reproduce it.
"""
import pytest
import torch

from gemma_csm.config import GemmaCSMConfig
from tests.test_model_cpu import tiny_backbone
from gemma_csm.model import GemmaCSM


@pytest.mark.slow
def test_overfit_one_clip_reproduces_codes():
    torch.manual_seed(0)
    cfg = GemmaCSMConfig(depth_d_model=64, depth_layers=2, depth_heads=2,
                         depth_loss_frame_frac=1.0)
    model = GemmaCSM(cfg, backbone=tiny_backbone())
    text = torch.randint(0, 500, (1, 5))
    codes = torch.randint(0, 2048, (1, 32, 8))
    batch = {
        "text_ids": text,
        "text_mask": torch.ones(1, 5, dtype=torch.bool),
        "codes": codes,
        "frame_mask": torch.ones(1, 9, dtype=torch.bool),  # 8 frames + EOS
    }
    batch["codes"] = torch.cat([codes, torch.zeros(1, 32, 1, dtype=torch.long)], -1)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(400):
        out = model(batch)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
    assert out["loss_c0"].item() < 0.2, f"c0 didn't memorize: {out['loss_c0']:.2f}"

    gen = model.generate_codes(text, max_frames=8, temperature=0.01)
    n = min(gen.shape[1], 8)
    fine_match = (gen[1:, :n] == codes[0, 1:, :n]).float().mean().item()
    assert fine_match > 0.9, (
        f"fine codebooks only {fine_match:.0%} reproduced after memorization "
        "-> depth decoder train/inference conditioning mismatch (leakage)"
    )
