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
    p.add_argument("--text", required=True, nargs="+",
                   help="one or more sentences; multiple write <out-stem>_NN.wav")
    p.add_argument("--out", type=Path, default=Path("sample.wav"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=250)
    # non-default depth decoder geometry (e.g. the m1_cap 425M: 1536/8/12)
    p.add_argument("--depth-d-model", type=int, default=None)
    p.add_argument("--depth-layers", type=int, default=None)
    p.add_argument("--depth-heads", type=int, default=None)
    # match the checkpoint's LoRA geometry (m2_sem: r32 alpha16 all-linear)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-all-linear", action="store_true")
    a = p.parse_args()

    cfg = GemmaCSMConfig()
    if a.lora_r:
        cfg.lora_r = a.lora_r
    if a.lora_alpha:
        cfg.lora_alpha = a.lora_alpha
    if a.lora_all_linear:
        cfg.lora_all_linear = True
    if a.depth_d_model:
        cfg.depth_d_model = a.depth_d_model
    if a.depth_layers:
        cfg.depth_layers = a.depth_layers
    if a.depth_heads:
        cfg.depth_heads = a.depth_heads
    model, tok = load_gemma_csm(cfg)
    sd = torch.load(a.ckpt, weights_only=True, map_location="cuda")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()

    codec = MimiCodec(device="cpu")
    for i, text in enumerate(a.text):
        ids = torch.tensor([tok(text)["input_ids"]], device="cuda")
        wav, codes, attempts = model.generate_wav(
            ids, codec, max_frames=a.max_frames,
            temperature=a.temperature, top_k=a.top_k)
        out = (a.out if len(a.text) == 1
               else a.out.with_name(f"{a.out.stem}_{i:02d}{a.out.suffix}"))
        sf.write(out, wav.float().numpy(), 24_000)
        retry_note = f", {attempts} attempts (silence-retry)" if attempts > 1 else ""
        print(f"wrote {out} ({codes.shape[1]} frames, "
              f"{codes.shape[1] / cfg.frame_rate:.1f}s{retry_note})")


if __name__ == "__main__":
    main()
