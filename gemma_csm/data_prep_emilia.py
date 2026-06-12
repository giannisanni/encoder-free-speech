"""Encode an Emilia-YODAS English subset to Mimi-code shards.

Emilia-YODAS is the CC BY 4.0 partition of amphion/Emilia-Dataset
(in-the-wild expressive speech; the plain "Emilia" partition is CC BY-NC
and must NOT be used — license constraint, see paper north stars).

Streams the webdataset tars from HuggingFace; each sample is an mp3 plus a
json with {text, speaker, duration, dnsmos, language}. We take English,
1-13 s, dnsmos >= --min-dnsmos, until --max-hours is reached.

NOTE: the dataset is gated on HF — accept the terms on the dataset page
once, and have a token in ~/.cache/huggingface/token (huggingface-cli login).

  python -m gemma_csm.data_prep_emilia --out data/shards_emilia \
      --max-hours 400 [--limit 50] [--device cuda]
"""
import argparse
import io
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from gemma_csm.codec import MimiCodec


def prepare(out_dir: Path, max_hours: float = 400.0, shard_size: int = 4000,
            limit: int | None = None, device: str = "cpu",
            max_seconds: float = 13.0, min_seconds: float = 1.0,
            min_dnsmos: float = 3.0):
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    ds = load_dataset("amphion/Emilia-Dataset",
                      data_files={"train": "Emilia-YODAS/EN/*.tar"},
                      split="train", streaming=True)
    items, n_shard, n_done, n_skipped, hours = [], 0, 0, 0, 0.0

    def flush():
        nonlocal items, n_shard
        if items:
            path = out_dir / f"emilia_yodas_{n_shard:03d}.pt"
            torch.save(items, path)
            print(f"wrote {path.name} ({len(items)} items, "
                  f"{hours:.1f}h total)", flush=True)
            items, n_shard = [], n_shard + 1

    pbar = tqdm(ds, desc="encoding", unit="utt")
    for ex in pbar:
        if limit and n_done >= limit:
            break
        if hours >= max_hours:
            break
        meta = ex["json"]
        if meta.get("language", "en") != "en":
            n_skipped += 1
            continue
        dur = float(meta.get("duration", 0))
        if not (min_seconds <= dur <= max_seconds):
            n_skipped += 1
            continue
        if float(meta.get("dnsmos", 0)) < min_dnsmos:
            n_skipped += 1
            continue
        raw = ex["mp3"]
        data, sr = sf.read(io.BytesIO(raw["bytes"] if isinstance(raw, dict)
                                      else raw), dtype="float32")
        if sr != 24_000:
            n_skipped += 1
            continue
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        items.append({
            "id": meta.get("id", str(n_done)),
            "speaker": str(meta.get("speaker", "unknown")),
            "text": (meta.get("text") or "").strip(),
            "codes": codec.encode(wav),
        })
        n_done += 1
        hours += dur / 3600
        pbar.set_postfix(hours=f"{hours:.1f}", skipped=n_skipped)
        if len(items) >= shard_size:
            flush()
    flush()
    print(f"done: {n_done} encoded ({hours:.1f}h), {n_skipped} skipped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=400.0)
    p.add_argument("--shard-size", type=int, default=4000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-dnsmos", type=float, default=3.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    prepare(a.out, a.max_hours, a.shard_size, a.limit, a.device,
            min_dnsmos=a.min_dnsmos)
