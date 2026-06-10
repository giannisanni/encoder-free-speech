"""Encode LJSpeech to Mimi codes once; training never touches raw audio.

Usage (substrate):
  python -m gemma_csm.data_prep --root data/LJSpeech-1.1 --out data/shards [--limit 50]
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from gemma_csm.codec import MimiCodec

LJ_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def prepare_shard(metadata_csv: Path, wav_dir: Path, out_dir: Path,
                  limit: int | None = None, device: str = "cpu") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    items = []
    lines = Path(metadata_csv).read_text().strip().split("\n")
    if limit:
        lines = lines[:limit]
    for line in tqdm(lines, desc="encoding"):
        clip_id, _raw, normalized = line.split("|", 2)
        data, sr = sf.read(str(Path(wav_dir) / f"{clip_id}.wav"), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim > 1:                      # stereo -> mono
            wav = wav.mean(-1)
        wav = torchaudio.functional.resample(wav, sr, 24_000)
        items.append({"id": clip_id, "text": normalized.strip(),
                      "codes": codec.encode(wav)})
    out = out_dir / f"ljspeech_{len(items)}.pt"
    torch.save(items, out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    prepare_shard(a.root / "metadata.csv", a.root / "wavs", a.out,
                  limit=a.limit, device=a.device)
