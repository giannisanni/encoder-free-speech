"""Encode LibriTTS-R to Mimi-code shards (multi-speaker, keeps speaker ids).

LibriTTS-R layout: <root>/<split>/<speaker>/<chapter>/<utt>.wav with a
matching <utt>.normalized.txt transcript per utterance. Audio is 24 kHz
(Mimi's native rate — no resampling needed).

Usage (substrate):
  python -m gemma_csm.data_prep_libritts --root data/LibriTTS_R/train-clean-360 \
      --out data/shards --shard-size 4000 [--limit 200]

Emits libritts_r_<n>.pt shards of items {id, speaker, text, codes}.
Speaker ids enable M1 voice-cloning pairs (same-speaker context+target).
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from gemma_csm.codec import MimiCodec


def iter_utterances(root: Path):
    for wav in sorted(root.rglob("*.wav")):
        txt = wav.with_suffix(".normalized.txt")
        if not txt.exists():
            txt = wav.with_suffix(".original.txt")
            if not txt.exists():
                continue
        speaker = wav.relative_to(root).parts[0]
        yield wav, txt, speaker


def prepare(root: Path, out_dir: Path, shard_size: int = 4000,
            limit: int | None = None, device: str = "cpu",
            max_seconds: float = 13.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    items, n_shard, n_done, n_skipped = [], 0, 0, 0

    def flush():
        nonlocal items, n_shard
        if items:
            path = out_dir / f"libritts_r_{n_shard:03d}.pt"
            torch.save(items, path)
            print(f"wrote {path.name} ({len(items)} items)", flush=True)
            items, n_shard = [], n_shard + 1

    utts = list(iter_utterances(root))
    if limit:
        utts = utts[:limit]
    for wav_path, txt_path, speaker in tqdm(utts, desc="encoding"):
        data, sr = sf.read(str(wav_path), dtype="float32")
        if sr != 24_000:  # LibriTTS-R is 24 kHz; guard anyway
            n_skipped += 1
            continue
        if len(data) > max_seconds * sr:  # keep clips within max_frames budget
            n_skipped += 1
            continue
        wav = torch.from_numpy(data)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        items.append({
            "id": wav_path.stem, "speaker": speaker,
            "text": txt_path.read_text(encoding="utf-8").strip(),
            "codes": codec.encode(wav),
        })
        n_done += 1
        if len(items) >= shard_size:
            flush()
    flush()
    print(f"done: {n_done} encoded, {n_skipped} skipped (rate/length)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=4000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    prepare(a.root, a.out, a.shard_size, a.limit, a.device)
