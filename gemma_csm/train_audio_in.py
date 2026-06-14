"""Train the 12B audio-IN (RQ1) model on OpenS2S speech-to-speech pairs.

Per item the sequence is [response_text | input-audio | response-audio frames]:
the frozen Gemma audio tower encodes the query speech (get_audio_features ->
audio_in_embeds) to provide UNDERSTANDING of how the input was said; LoRA on the
text decoder + our speech-out heads (c0 + depth) are trained to produce the
response speech. QLoRA (4-bit NF4 + LoRA) + PagedAdamW8bit + batch-1/grad-accum
+ sequence caps for the 16GB edge. A held-out val split (--val-frac) prints
`val N c0 X depth Y` so log_to_tb surfaces val/c0_semantic on TensorBoard.

Run (audio-IN venv):
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 ~/audioin-venv/bin/python \\
      -m gemma_csm.train_audio_in --shard 'data/shards_opens2s/*.pt' \\
      --steps 60000 --out checkpoints/a1_audioin --grad-accum 16 \\
      --max-audio 64 --max-frames 150 --eval-every 2000 --val-every 2000
"""
import argparse
import glob
import os
import time
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset_opens2s import OpenS2SDataset, collate_opens2s
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.train import trainable_state_dict


def make_paged_opt(model, lr, depth_lr_mult):
    """PagedAdamW8bit (QLoRA-standard): ~4x smaller state than fp32 + CPU paging
    under pressure -- needed at the 12B 16GB edge. Depth decoder in its own group."""
    import bitsandbytes as bnb
    depth, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (depth if name.startswith("depth.") else rest).append(p)
    return bnb.optim.PagedAdamW8bit([
        {"params": rest, "lr": lr},
        {"params": depth, "lr": lr * depth_lr_mult},
    ])


@torch.no_grad()
def evaluate(model, val_items, audio_encode, tok, max_audio, max_frames,
             max_n=64, no_audio_in=False, max_text=96):
    """Mean held-out c0/depth (overfitting sentinel). batch-1 to stay in VRAM."""
    model.eval()
    c0s, dps = [], []
    for it in val_items[:max_n]:
        b = collate_opens2s([it], tok, max_audio, max_frames, max_text)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        if not no_audio_in:
            b["audio_in_embeds"] = audio_encode(
                b["input_features"].to(torch.bfloat16), b["input_features_mask"])
            b["audio_in_mask"] = b["input_features_mask"]
        out = model(b)
        c0s.append(float(out["loss_c0"]))
        dps.append(float(out["loss_depth"]))
    model.train()
    return (sum(c0s) / len(c0s), sum(dps) / len(dps)) if c0s else (float("nan"), float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--depth-frac", type=float, default=0.25)
    p.add_argument("--depth-lr-mult", type=float, default=2.0)
    p.add_argument("--depth-weight", type=float, default=0.01)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-all-linear", action="store_true")
    p.add_argument("--max-audio", type=int, default=64)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--max-text", type=int, default=96)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--backbone-id", default="google/gemma-4-12B-it")
    p.add_argument("--no-audio-in", action="store_true",
                   help="Phase A: text->audio only (skip audio tower); heads learn to speak")
    p.add_argument("--sample-every", type=int, default=0,
                   help="every N steps decode a few clips in-process (0=off)")
    p.add_argument("--sample-clips", type=int, default=3)
    p.add_argument("--sample-frames", type=int, default=150)
    a = p.parse_args()

    cfg = GemmaCSMConfig()
    cfg.backbone_id = a.backbone_id
    cfg.lora_r, cfg.lora_alpha = a.lora_r, a.lora_alpha
    cfg.lora_all_linear = a.lora_all_linear
    cfg.depth_loss_frame_frac = a.depth_frac
    cfg.depth_loss_weight = a.depth_weight

    model, tok, audio_encode = load_gemma_csm_audio_in(cfg)

    start_step = 0
    resume_opt = None
    if a.resume:
        ckpt = torch.load(a.resume, weights_only=True, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and "step" in ckpt:
            sd, resume_opt, start_step = ckpt["model"], ckpt.get("opt"), int(ckpt["step"])
        else:
            sd = ckpt
            mm = re.search(r"(\d+)", Path(a.resume).stem)
            start_step = int(mm.group(1)) if mm else 0
        own = model.state_dict()
        for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
            del sd[k]
        model.load_state_dict(sd, strict=False)
        del sd, own
        torch.cuda.empty_cache()
        print(f"resumed from {a.resume} at step {start_step} "
              f"(optimizer {'restored' if resume_opt is not None else 'fresh'})", flush=True)

    shards = []
    for s in a.shard:
        shards += glob.glob(s)
    ds = OpenS2SDataset(sorted(shards))
    n_val = max(1, int(len(ds) * a.val_frac)) if len(ds) > 20 else 0
    val_items = ds.items[len(ds) - n_val:] if n_val else []
    ds.items = ds.items[:len(ds) - n_val] if n_val else ds.items
    print(f"dataset: {len(ds)} train / {len(val_items)} val pairs", flush=True)

    dl = DataLoader(
        ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_opens2s(b, tok, a.max_audio, a.max_frames, a.max_text),
    )
    opt = make_paged_opt(model, cfg.lr, a.depth_lr_mult)
    if resume_opt is not None:
        try:
            opt.load_state_dict(resume_opt)
            print("optimizer state restored", flush=True)
        except Exception as e:
            print(f"WARN optimizer restore failed ({e}); continuing fresh", flush=True)
    del resume_opt
    a.out.mkdir(parents=True, exist_ok=True)

    def save_ckpt(step):
        dst = a.out / f"ckpt_{step}.pt"
        tmp = a.out / f"ckpt_{step}.pt.tmp"
        torch.save({"model": trainable_state_dict(model),
                    "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, dst)
        if a.keep_ckpts:
            by = sorted(a.out.glob("ckpt_*.pt"),
                        key=lambda x: int(x.stem.split("_")[1]))
            for old in by[:-a.keep_ckpts]:
                old.unlink()
        print(f"saved ckpt_{step}", flush=True)

    _codec = {}
    def do_sample(step):
        if not a.sample_every or step % a.sample_every != 0:
            return
        import torchaudio
        from gemma_csm.codec import MimiCodec
        if "c" not in _codec:
            _codec["c"] = MimiCodec(device="cuda")
        codec = _codec["c"]
        sdir = a.out / "samples"; sdir.mkdir(parents=True, exist_ok=True)
        pool = [it.get("text", "").strip() for it in (val_items or ds.items)]
        prompts = [t for t in pool if 20 <= len(t) <= 160][:a.sample_clips]
        model.eval(); torch.cuda.empty_cache()
        try:
            for i, text in enumerate(prompts):
                ids = torch.tensor([tok(text)["input_ids"]], device="cuda")
                wav, codes, att = model.generate_wav(
                    ids, codec, max_frames=a.sample_frames, temperature=0.9, top_k=50)
                torchaudio.save(str(sdir / f"step{step:06d}_s{i}.wav"),
                                wav.detach().cpu().unsqueeze(0), 24000)
                print(f"sample step{step} s{i} {wav.numel()/24000:.1f}s att={att}", flush=True)
        except RuntimeError as e:
            print(f"WARN sampling skipped (likely VRAM): {e}", flush=True)
        finally:
            model.train(); torch.cuda.empty_cache()

    step, t0 = start_step, time.time()
    while step < a.steps:
        for batch in dl:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if not a.no_audio_in:
                batch["audio_in_embeds"] = audio_encode(
                    batch["input_features"].to(torch.bfloat16),
                    batch["input_features_mask"])
                batch["audio_in_mask"] = batch["input_features_mask"]
            out = model(batch)
            (out["loss"] / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            if step % 50 == 0:
                print(f"step {step}  c0 {out['loss_c0']:.3f}  "
                      f"depth {out['loss_depth']:.3f}  "
                      f"{(time.time() - t0) / max(step - start_step, 1):.2f}s/step", flush=True)
            step += 1
            if val_items and step % a.val_every == 0:
                vc0, vd = evaluate(model, val_items, audio_encode, tok,
                                   a.max_audio, a.max_frames, no_audio_in=a.no_audio_in, max_text=a.max_text)
                print(f"val {step}  c0 {vc0:.3f}  depth {vd:.3f}", flush=True)
            if step % a.eval_every == 0 or step >= a.steps:
                save_ckpt(step)
                do_sample(step)
            if step >= a.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
