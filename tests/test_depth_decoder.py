import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import DepthDecoder


def test_depth_decoder_shapes_and_loss():
    cfg = GemmaCSMConfig()
    dd = DepthDecoder(cfg, backbone_dim=512)
    n_frames = 6
    h = torch.randn(n_frames, 512)                       # backbone hidden per frame
    codes = torch.randint(0, 2048, (n_frames, 32))       # full frame codes
    loss = dd.loss(h, codes)
    assert loss.ndim == 0 and loss.item() > 0


def test_depth_decoder_generate():
    cfg = GemmaCSMConfig()
    dd = DepthDecoder(cfg, backbone_dim=512)
    h = torch.randn(1, 512)
    c0 = torch.randint(0, 2048, (1,))
    frame = dd.generate(h, c0)                           # (1, 32)
    assert frame.shape == (1, 32)
    assert (frame[:, 0] == c0).all()
