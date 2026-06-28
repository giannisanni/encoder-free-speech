"""CPU unit test for GemmaCSM.forward_unified geometry/indexing/numerics.

Like test_forward_joint but for the native-delivery path: the query audio is
wrapped in the real chat-template framing tokens (ids up to 258883), so the fake
vocab must cover them. Validates losses + grads are finite on a tiny fake backbone.
Run: python tests/test_forward_unified.py
"""
import torch
import torch.nn as nn
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM

VOCAB = 259000  # must exceed the framing token ids (max 258883)


class _Cfg:
    hidden_size = 64
    final_logit_softcapping = None


class FakeBackbone(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.block = nn.Linear(d, d)
        self.config = _Cfg()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, output_hidden_states=False, **kw):
        h = self.block(inputs_embeds)
        class O:  # noqa
            pass
        o = O(); o.hidden_states = (inputs_embeds, h); return o


def make_batch(B=2, Tt=10, Tf=9, Ta=6, d=64, codebook=2048):
    text_ids = torch.randint(0, 500, (B, Tt))
    text_mask = torch.zeros(B, Tt, dtype=torch.bool)
    codes = torch.randint(0, codebook, (B, 32, Tf))
    frame_mask = torch.zeros(B, Tf, dtype=torch.bool)
    ain = torch.randn(B, Ta, d)
    ain_mask = torch.zeros(B, Ta, dtype=torch.bool)
    lens = [(8, 7, 5), (4, 3, 6)]
    for i, (li, fi, ai) in enumerate(lens[:B]):
        text_mask[i, :li] = True
        frame_mask[i, :fi + 1] = True
        ain_mask[i, :ai] = True
    return {"text_ids": text_ids, "text_mask": text_mask, "codes": codes,
            "frame_mask": frame_mask, "audio_in_embeds": ain, "audio_in_mask": ain_mask}


def main():
    torch.manual_seed(0)
    cfg = GemmaCSMConfig()
    d = _Cfg.hidden_size
    model = GemmaCSM(cfg, FakeBackbone(d))
    print("uses_ple:", model.uses_ple, "(want False)")
    batch = make_batch(d=d, codebook=cfg.codebook_size)
    out = model.forward_unified(batch, eos_id=1)
    for k in ("loss", "loss_text", "loss_c0", "loss_depth"):
        print(f"  {k} = {float(out[k]):.4f}  finite={torch.isfinite(out[k]).item()}")
        assert torch.isfinite(out[k]).all(), f"{k} not finite!"
    out["loss"].backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        print("NON-FINITE GRADS in:", bad[:20]); raise SystemExit("FAIL")
    print("ALL FINITE (forward + backward). forward_unified geometry OK on fake backbone.")


if __name__ == "__main__":
    main()
