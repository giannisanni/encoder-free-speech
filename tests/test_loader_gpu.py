import pytest
import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm


@pytest.mark.gpu
def test_load_and_forward_fits_16gb():
    model, tok = load_gemma_csm(GemmaCSMConfig())
    ids = torch.tensor([tok("Hello world.")["input_ids"]], device="cuda")
    batch = {
        "text_ids": ids,
        "text_mask": torch.ones_like(ids, dtype=torch.bool),
        "codes": torch.randint(0, 2048, (1, 32, 25), device="cuda"),
        "frame_mask": torch.ones(1, 25, dtype=torch.bool, device="cuda"),
    }
    out = model(batch)
    out["loss"].backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {peak_gb:.1f} GB")
    assert out["loss"].item() > 0
    assert peak_gb < 14.0
