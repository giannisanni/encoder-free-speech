import pytest
import torch

from gemma_csm.codec import MimiCodec


@pytest.fixture(scope="module")
def codec():
    return MimiCodec(device="cpu")


def test_encode_shape_and_rate(codec):
    wav = torch.sin(2 * torch.pi * 220 * torch.arange(24_000) / 24_000)  # 1 s tone
    codes = codec.encode(wav)
    assert codes.shape[0] == 32                  # 32 codebooks
    assert 11 <= codes.shape[1] <= 14            # ~12.5 frames per second
    assert codes.dtype == torch.long
    assert codes.max() < 2048 and codes.min() >= 0


def test_roundtrip_produces_audio(codec):
    wav = torch.sin(2 * torch.pi * 220 * torch.arange(24_000) / 24_000)
    out = codec.decode(codec.encode(wav))
    assert out.ndim == 1
    assert abs(out.shape[0] - 24_000) < 4_000    # roughly 1 s back
    assert out.abs().max() > 0.01                # not silence
