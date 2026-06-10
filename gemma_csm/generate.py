"""Generate speech from a trained checkpoint.

  python -m gemma_csm.generate --ckpt checkpoints/m0a/ckpt_6000.pt \
      --text "Printing, in the only sense..." --temperature 0.6 --top-k 50 \
      --out sample.wav
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", type=Path, default=Path("sample.wav"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=250)
    a = p.parse_args()

    cfg = GemmaCSMConfig()
    model, tok = load_gemma_csm(cfg)
    sd = torch.load(a.ckpt, weights_only=True, map_location="cuda")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()

    ids = torch.tensor([tok(a.text)["input_ids"]], device="cuda")
    codes = model.generate_codes(ids, max_frames=a.max_frames,
                                 temperature=a.temperature, top_k=a.top_k)
    wav = MimiCodec(device="cpu").decode(codes)
    sf.write(a.out, wav.float().numpy(), 24_000)
    print(f"wrote {a.out} ({codes.shape[1]} frames, "
          f"{codes.shape[1] / cfg.frame_rate:.1f}s)")


if __name__ == "__main__":
    main()
