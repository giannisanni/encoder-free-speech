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
import functools
import os
import time
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset_opens2s import OpenS2SDataset, collate_opens2s
from gemma_csm.dataset import CodesDataset, collate as collate_tts
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.train import trainable_state_dict


def make_paged_opt(model, lr, depth_lr_mult, full=False):
    """Optimizer with the depth decoder in its own (higher-LR) group.
    full=True -> full fp32 AdamW (stable; ~3GB state for our trainable params,
    fits 48GB; use on RunPod). full=False -> PagedAdamW8bit (8-bit state, needed
    only at the 12B 16GB substrate edge -- but its quantized-variance m/sqrt(v)
    blow-up NaNs the joint run, so prefer full=True off-substrate)."""
    depth, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (depth if name.startswith("depth.") else rest).append(p)
    groups = [{"params": rest, "lr": lr}, {"params": depth, "lr": lr * depth_lr_mult}]
    if full:
        return torch.optim.AdamW(groups)
    import bitsandbytes as bnb
    return bnb.optim.PagedAdamW8bit(groups, eps=1e-6)


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


@torch.no_grad()
def evaluate_joint(model, val_items, audio_encode, tok, eos_id, max_audio, max_frames,
                   max_n=48, max_text=96):
    """Teacher-forced held-out text/c0/depth for --joint (overfit sentinel). The
    real quality gate is a text-free decode (generate_joint); this just tracks
    that the three losses generalise."""
    model.eval()
    txt, c0s, dps = [], [], []
    for it in val_items[:max_n]:
        b = collate_opens2s([it], tok, max_audio, max_frames, max_text)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        b["audio_in_embeds"] = audio_encode(
            b["input_features"].to(torch.bfloat16), b["input_features_mask"])
        b["audio_in_mask"] = b["input_features_mask"]
        out = model.forward_joint(b, eos_id, tok.bos_token_id)
        txt.append(float(out["loss_text"])); c0s.append(float(out["loss_c0"])); dps.append(float(out["loss_depth"]))
    model.train()
    n = len(c0s) or 1
    return sum(txt) / n, sum(c0s) / n, sum(dps) / n


def evaluate_unified(model, val_items, audio_encode, tok, eos_id, max_audio, max_frames,
                     max_n=48, max_text=96):
    """Teacher-forced held-out text/c0/depth for --unified (native-delivery joint).
    Same sentinel role as evaluate_joint; the real gate is a free-running decode."""
    model.eval()
    txt, c0s, dps = [], [], []
    for it in val_items[:max_n]:
        b = collate_opens2s([it], tok, max_audio, max_frames, max_text)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        # native delivery: forward_unified reads raw input_features (no pre-splice)
        out = model.forward_unified(b, eos_id)
        txt.append(float(out["loss_text"])); c0s.append(float(out["loss_c0"])); dps.append(float(out["loss_depth"]))
    model.train()
    n = len(c0s) or 1
    return sum(txt) / n, sum(c0s) / n, sum(dps) / n


def evaluate_grounded(model, val_items, audio_encode, tok, eos_id, max_audio, max_frames,
                      max_n=48, max_text=96):
    """Teacher-forced held-out comprehension/text/c0/depth for --grounded. The real
    gate is transcript WER from a generate_grounded decode; this tracks the losses."""
    model.eval()
    cmp, txt, c0s, dps = [], [], [], []
    for it in val_items[:max_n]:
        b = collate_opens2s([it], tok, max_audio, max_frames, max_text)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        b["audio_in_embeds"] = audio_encode(
            b["input_features"].to(torch.bfloat16), b["input_features_mask"])
        b["audio_in_mask"] = b["input_features_mask"]
        out = model.forward_grounded(b, eos_id, tok.bos_token_id)
        cmp.append(float(out["loss_compr"])); txt.append(float(out["loss_text"]))
        c0s.append(float(out["loss_c0"])); dps.append(float(out["loss_depth"]))
    model.train()
    n = len(c0s) or 1
    return sum(cmp) / n, sum(txt) / n, sum(c0s) / n, sum(dps) / n


def evaluate_tts(model, val_raw, tok, pad_id, max_n=64):
    """Text->speech held-out c0/depth for TTS (CodesDataset) shards."""
    model.eval()
    c0s, dps = [], []
    for it in val_raw[:max_n]:
        text = f"[{it.get('speaker', 'lj')}] {it['text']}"
        ids = torch.tensor(tok(text, add_special_tokens=True)["input_ids"], dtype=torch.long)
        b = collate_tts([{"text_ids": ids, "codes": it["codes"]}], pad_id)
        b = {k: v.cuda() for k, v in b.items()}
        out = model(b)
        c0s.append(float(out["loss_c0"]))
        dps.append(float(out["loss_depth"]))
    model.train()
    if not c0s:
        return float("nan"), float("nan")
    return sum(c0s) / len(c0s), sum(dps) / len(dps)


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
    p.add_argument("--lora-coverage", default="", choices=["", "attn", "attn_down", "all"],
                   help="unified loader coverage tier (overrides --lora-all-linear when set)")
    p.add_argument("--max-audio", type=int, default=64)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--max-text", type=int, default=96)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="override config lr (big-batch)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0=off)")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear LR warmup from 0 over N steps (anti early-blowup; needed for --joint)")
    p.add_argument("--full-opt", action="store_true",
                   help="full fp32 AdamW instead of PagedAdamW8bit (stable; needs ~3GB more, use on 48GB GPUs)")
    p.add_argument("--backbone-id", default="google/gemma-4-12B-it")
    p.add_argument("--no-audio-in", action="store_true",
                   help="Phase A: text->audio only (skip audio tower); heads learn to speak")
    p.add_argument("--tts", action="store_true",
                   help="train on TTS shards (CodesDataset text->speech); implies --no-audio-in")
    p.add_argument("--joint", action="store_true",
                   help="joint text+audio S2S: [query-audio | response-text | response-audio]; "
                        "model GENERATES the reply text (LM loss) then speaks it. Requires audio-in.")
    p.add_argument("--text-weight", type=float, default=1.0,
                   help="weight on the response-text LM loss in --joint mode")
    p.add_argument("--grounded", action="store_true",
                   help="grounded inner monologue: [query-audio | query-transcript | reply-text | "
                        "reply-audio]; model TRANSCRIBES then replies then speaks. Requires audio-in.")
    p.add_argument("--comprehension-weight", type=float, default=1.0,
                   help="weight on the query-transcript (comprehension) LM loss in --grounded mode")
    p.add_argument("--unified", action="store_true",
                   help="native-delivery joint S2S: query audio wrapped in the real chat-template "
                        "framing (carries CONTENT, not just prosody) | response-text | response-audio. "
                        "Reuses --text-weight. Requires audio-in.")
    p.add_argument("--sample-every", type=int, default=0,
                   help="every N steps decode a few clips in-process (0=off)")
    p.add_argument("--sample-clips", type=int, default=3)
    p.add_argument("--sample-frames", type=int, default=150)
    a = p.parse_args()

    cfg = GemmaCSMConfig()
    cfg.backbone_id = a.backbone_id
    cfg.lora_r, cfg.lora_alpha = a.lora_r, a.lora_alpha
    cfg.lora_all_linear = a.lora_all_linear
    cfg.lora_coverage = a.lora_coverage
    cfg.depth_loss_frame_frac = a.depth_frac
    cfg.depth_loss_weight = a.depth_weight
    cfg.text_loss_weight = a.text_weight
    cfg.comprehension_loss_weight = a.comprehension_weight
    if (a.joint or a.grounded or a.unified) and (a.tts or a.no_audio_in):
        raise SystemExit("--joint/--grounded need audio-in: do not pass --tts/--no-audio-in")

    model, tok, audio_encode = load_gemma_csm_audio_in(cfg)

    if a.unified:
        # Native delivery (forward_unified) runs the FULL model with use_cache=True
        # (seg1) then hands the KV to the backbone (seg2). Gradient checkpointing
        # forces use_cache=False, so it must be OFF. Memory budget: 13.4 GB at
        # batch 1 (validated) -> force batch 1; use --grad-accum for effective batch.
        for m in (getattr(model, "full_model", None), model.backbone):
            try:
                m.gradient_checkpointing_disable()
            except Exception as e:
                print("gc disable note:", e, flush=True)
        if a.batch_size != 1:
            print(f"--unified forces batch-size 1 (was {a.batch_size}); use --grad-accum", flush=True)
            a.batch_size = 1

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
    shards = sorted(shards)
    if a.tts:
        a.no_audio_in = True
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        ds = CodesDataset(shards, tok, max_frames=a.max_frames)
        n_val = max(1, int(len(ds) * a.val_frac)) if len(ds) > 20 else 0
        val_items = ds.items[len(ds) - n_val:] if n_val else []
        ds.items = ds.items[:len(ds) - n_val] if n_val else ds.items
        print(f"TTS dataset: {len(ds)} train / {len(val_items)} val (pad_id={pad_id})", flush=True)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        num_workers=a.num_workers,
                        collate_fn=functools.partial(collate_tts, text_pad_id=pad_id))
    else:
        pad_id = None
        ds = OpenS2SDataset(shards)
        n_val = max(1, int(len(ds) * a.val_frac)) if len(ds) > 20 else 0
        val_items = ds.items[len(ds) - n_val:] if n_val else []
        ds.items = ds.items[:len(ds) - n_val] if n_val else ds.items
        print(f"dataset: {len(ds)} train / {len(val_items)} val pairs", flush=True)
        dl = DataLoader(
            ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
            collate_fn=functools.partial(collate_opens2s, tok=tok, max_audio=a.max_audio,
                                         max_frames=a.max_frames, max_text=a.max_text),
        )
    opt = make_paged_opt(model, a.lr if a.lr is not None else cfg.lr, a.depth_lr_mult, full=a.full_opt)
    if resume_opt is not None:
        try:
            opt.load_state_dict(resume_opt)
            _base = a.lr if a.lr is not None else cfg.lr
            opt.param_groups[0]["lr"] = _base
            opt.param_groups[1]["lr"] = _base * a.depth_lr_mult
            print(f"optimizer state restored; LR override rest={_base} depth={_base*a.depth_lr_mult}", flush=True)
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
        _src = val_items or ds.items
        if a.tts:
            pool = [f"[{it.get('speaker', 'lj')}] {it['text']}".strip() for it in _src]
        else:
            pool = [it.get("text", "").strip() for it in _src]
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

    if os.environ.get("JOINT_ANOMALY"):  # localize finite-forward/NaN-backward op
        torch.autograd.set_detect_anomaly(True)
        print("autograd anomaly detection ON", flush=True)
    step, t0 = start_step, time.time()
    while step < a.steps:
        for batch in dl:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if not a.no_audio_in and not a.unified:
                # --unified delivers audio NATIVELY inside forward_unified (full
                # model + input_features); it must NOT receive a pre-encoded splice.
                batch["audio_in_embeds"] = audio_encode(
                    batch["input_features"].to(torch.bfloat16),
                    batch["input_features_mask"])
                batch["audio_in_mask"] = batch["input_features_mask"]
            if a.grounded:
                out = model.forward_grounded(batch, tok.eos_token_id, tok.bos_token_id)
            elif a.joint:
                out = model.forward_joint(batch, tok.eos_token_id, tok.bos_token_id)
            elif a.unified:
                out = model.forward_unified(batch, tok.eos_token_id)
            else:
                out = model(batch)
            if not torch.isfinite(out["loss"]):
                print(f"WARN non-finite loss at step {step} "
                      f"(text={float(out.get('loss_text', 0)):.2f} c0={float(out['loss_c0']):.2f} "
                      f"depth={float(out['loss_depth']):.2f}); skipping step", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            (out["loss"] / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0:
                if a.warmup_steps and step < a.warmup_steps:
                    scale = (step + 1) / a.warmup_steps
                    _base = a.lr if a.lr is not None else cfg.lr
                    opt.param_groups[0]["lr"] = _base * scale
                    opt.param_groups[1]["lr"] = _base * a.depth_lr_mult * scale
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for grp in opt.param_groups for p in grp["params"]], a.grad_clip)
                # GRAD GUARD: a finite loss can still yield a NaN/Inf gradient in
                # backward; clip_grad_norm_ then spreads that NaN to EVERY param via
                # the global norm, so the first opt.step poisons all weights. Skip
                # the step when the norm is non-finite (weights stay clean).
                if not torch.isfinite(gnorm):
                    bad = "?"
                    if os.environ.get("JOINT_GRAD_DEBUG"):  # localize the source param
                        for _n, _p in model.named_parameters():
                            if _p.grad is not None and not torch.isfinite(_p.grad).all():
                                bad = _n
                                break
                    print(f"WARN non-finite grad-norm at step {step} "
                          f"(norm={float(gnorm):.3e}, first-bad-param={bad}); skipping opt.step",
                          flush=True)
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    continue
                opt.step()
                opt.zero_grad()
            if step % 50 == 0:
                cx = f"compr {out['loss_compr']:.3f}  " if "loss_compr" in out else ""
                tx = f"text {out['loss_text']:.3f}  " if "loss_text" in out else ""
                print(f"step {step}  {cx}{tx}c0 {out['loss_c0']:.3f}  "
                      f"depth {out['loss_depth']:.3f}  "
                      f"{(time.time() - t0) / max(step - start_step, 1):.2f}s/step", flush=True)
            step += 1
            if val_items and step % a.val_every == 0:
                if a.grounded:
                    vcmp, vt, vc0, vd = evaluate_grounded(model, val_items, audio_encode, tok,
                                                          tok.eos_token_id, a.max_audio, a.max_frames,
                                                          max_text=a.max_text)
                    print(f"val {step}  compr {vcmp:.3f}  text {vt:.3f}  c0 {vc0:.3f}  depth {vd:.3f}",
                          flush=True)
                elif a.joint:
                    vt, vc0, vd = evaluate_joint(model, val_items, audio_encode, tok,
                                                 tok.eos_token_id, a.max_audio, a.max_frames,
                                                 max_text=a.max_text)
                    print(f"val {step}  text {vt:.3f}  c0 {vc0:.3f}  depth {vd:.3f}", flush=True)
                elif a.unified:
                    vt, vc0, vd = evaluate_unified(model, val_items, audio_encode, tok,
                                                   tok.eos_token_id, a.max_audio, a.max_frames,
                                                   max_text=a.max_text)
                    print(f"val {step}  text {vt:.3f}  c0 {vc0:.3f}  depth {vd:.3f}", flush=True)
                elif a.tts:
                    vc0, vd = evaluate_tts(model, val_items, tok, pad_id)
                    print(f"val {step}  c0 {vc0:.3f}  depth {vd:.3f}", flush=True)
                else:
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
