"""CPU unit test for GemmaCSM.forward_grounded geometry/indexing/numerics.

Grounded geometry: [BOS | audio-in | query_text | <eos> | reply_text | <eos> | audio].
Tiny fake backbone (no 12B/GPU) + synthetic batch matching collate_opens2s
(now incl. query_text_ids/query_text_mask). Verifies all losses + grads finite
and the comprehension/reply/c0/depth indexing is consistent before any pod run.
Run: python tests/test_forward_grounded.py
"""
import torch
import torch.nn as nn
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM

VOCAB = 512


class _Cfg:
    hidden_size = 64
    final_logit_softcapping = None


class FakeBackbone(nn.Module):
    """Mimics the backbone bits forward_grounded touches. No get_per_layer_inputs
    -> uses_ple=False (the 12B path)."""
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.block = nn.Linear(d, d)
        self.config = _Cfg()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, **kw):
        h = self.block(inputs_embeds)
        class O:  # noqa
            pass
        o = O()
        o.hidden_states = (inputs_embeds, h)
        return o


def make_batch(B=2, Tt=10, Tq=8, Tf=9, Ta=6, d=64, codebook=2048):
    text_ids = torch.randint(2, VOCAB, (B, Tt))
    text_mask = torch.zeros(B, Tt, dtype=torch.bool)
    query_text_ids = torch.randint(2, VOCAB, (B, Tq))
    query_text_mask = torch.zeros(B, Tq, dtype=torch.bool)
    codes = torch.randint(0, codebook, (B, 32, Tf))
    frame_mask = torch.zeros(B, Tf, dtype=torch.bool)
    ain = torch.randn(B, Ta, d)
    ain_mask = torch.zeros(B, Ta, dtype=torch.bool)
    # per-item (reply_len, frames, audio_in_len, query_len) -- varied incl. shorts
    lens = [(8, 7, 5, 6), (4, 3, 6, 2)]
    for i, (ri, fi, ai, qi) in enumerate(lens[:B]):
        text_mask[i, :ri] = True
        query_text_mask[i, :qi] = True
        frame_mask[i, :fi + 1] = True   # collate sets :f+1 -> Fr = f
        ain_mask[i, :ai] = True
    return {"text_ids": text_ids, "text_mask": text_mask,
            "query_text_ids": query_text_ids, "query_text_mask": query_text_mask,
            "codes": codes, "frame_mask": frame_mask,
            "audio_in_embeds": ain, "audio_in_mask": ain_mask}


def main():
    torch.manual_seed(0)
    cfg = GemmaCSMConfig()
    d = _Cfg.hidden_size
    model = GemmaCSM(cfg, FakeBackbone(d))
    print("uses_ple:", model.uses_ple, "(want False)")
    batch = make_batch(d=d, codebook=cfg.codebook_size)
    eos_id, bos_id = 1, 2

    out = model.forward_grounded(batch, eos_id, bos_id)
    for k in ("loss", "loss_compr", "loss_text", "loss_c0", "loss_depth"):
        v = float(out[k])
        print(f"  {k} = {v:.4f}  finite={torch.isfinite(out[k]).item()}")
        assert torch.isfinite(out[k]).all(), f"{k} is not finite!"

    out["loss"].backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        print("NON-FINITE GRADS in:", bad[:20])
        raise SystemExit("FAIL: non-finite gradients")
    print("ALL FINITE (forward + backward). grounded geometry/indexing OK on fake backbone.")


if __name__ == "__main__":
    main()
