import torch

from gemma_csm.train import make_optimizer
from tests.test_model_cpu import make_model


def test_depth_param_group_gets_multiplied_lr():
    model, cfg = make_model()
    opt = make_optimizer(model, lr=1e-4, depth_lr_mult=2.0)
    assert len(opt.param_groups) == 2
    rest, depth = opt.param_groups
    assert depth["lr"] == 2e-4 and rest["lr"] == 1e-4
    n_depth = sum(p.numel() for p in depth["params"])
    n_model_depth = sum(p.numel() for p in model.depth.parameters())
    assert n_depth == n_model_depth          # exactly the depth decoder
    # every trainable param is in exactly one group
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_rest = sum(p.numel() for p in rest["params"])
    assert n_rest + n_depth == n_total
