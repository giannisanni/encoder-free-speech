"""An item's c0 predictions must not depend on what it's batched with.

With right-padded text, padding lands BETWEEN text and audio, shifting all
audio positions for short-text items depending on batch pairing — training
geometry then mismatches inference (which has no padding).
"""
import torch

from tests.test_model_cpu import make_model


def _item(text_len, n_frames, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 500, (text_len,), generator=g),
            torch.randint(0, 2048, (32, n_frames), generator=g))


def _batch(items):
    tmax = max(len(t) for t, _ in items)
    fmax = max(c.shape[1] for _, c in items) + 1
    B = len(items)
    out = {
        "text_ids": torch.zeros(B, tmax, dtype=torch.long),
        "text_mask": torch.zeros(B, tmax, dtype=torch.bool),
        "codes": torch.zeros(B, 32, fmax, dtype=torch.long),
        "frame_mask": torch.zeros(B, fmax, dtype=torch.bool),
    }
    for i, (t, c) in enumerate(items):
        out["text_ids"][i, : len(t)] = t
        out["text_mask"][i, : len(t)] = True
        out["codes"][i, :, : c.shape[1]] = c
        out["frame_mask"][i, : c.shape[1] + 1] = True
    return out


def test_c0_logits_invariant_to_batch_padding():
    torch.manual_seed(0)
    model, cfg = make_model()
    model.eval()
    short = _item(text_len=3, n_frames=5, seed=1)
    long = _item(text_len=9, n_frames=12, seed=2)

    with torch.no_grad():
        alone = model(_batch([short]))["c0_logits"][0]      # (6, V)
        mixed = model(_batch([short, long]))["c0_logits"][0]  # padded; first 6 rows real

    n = short[1].shape[1] + 1  # frames + EOS prediction
    assert torch.allclose(alone[:n], mixed[:n], atol=1e-4), (
        "c0 predictions for an item change when batched with a longer item "
        "-> padding sits between text and audio, shifting training geometry"
    )
