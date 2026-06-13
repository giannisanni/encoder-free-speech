"""Encode LibriTTS-R to Mimi-code shards (multi-speaker, keeps speaker ids).

LibriTTS-R layout: <root>/<split>/<speaker>/<chapter>/<utt>.wav with a
matching <utt>.normalized.txt transcript per utterance. Audio is 24 kHz
(Mimi's native rate — no resampling needed).

Usage (substrate), from an extracted directory:
  python -m gemma_csm.data_prep_libritts --root data/LibriTTS_R/train-clean-360 \
      --out data/shards --shard-size 4000 [--limit 200]

Or streaming straight out of the tar.gz — wavs are decoded in memory and
NEVER extracted to disk (the full extraction would need ~80 GB):
  python -m gemma_csm.data_prep_libritts --from-tar data/train_clean_360.tar.gz \
      --out data/shards

Emits libritts_r_<n>.pt shards of items {id, speaker, text, codes}.
Speaker ids enable M1 voice-cloning pairs (same-speaker context+target).
"""
import argparse
import io
import tarfile
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


def prepare_from_tar(tar_path: Path, out_dir: Path, shard_size: int = 4000,
                     limit: int | None = None, device: str = "cpu",
                     max_seconds: float = 13.0):
    """Single sequential pass over the tar.gz; wavs decoded in memory only.

    Transcripts (.normalized.txt) appear near their wavs in the archive; we
    buffer whichever of the pair arrives first and emit when both are seen.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    items, n_shard, n_done, n_skipped = [], 0, 0, 0
    texts: dict[str, str] = {}
    wavs: dict[str, tuple] = {}

    def flush():
        nonlocal items, n_shard
        if items:
            path = out_dir / f"libritts_r_{n_shard:03d}.pt"
            torch.save(items, path)
            print(f"wrote {path.name} ({len(items)} items)", flush=True)
            items, n_shard = [], n_shard + 1

    def emit(stem: str):
        nonlocal n_done, n_skipped
        data, speaker = wavs.pop(stem)
        text = texts.pop(stem)
        if len(data) > max_seconds * 24_000:
            n_skipped += 1
            return
        wav = torch.from_numpy(data)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        items.append({"id": stem, "speaker": speaker, "text": text,
                      "codes": codec.encode(wav)})
        n_done += 1
        if len(items) >= shard_size:
            flush()

    with tarfile.open(tar_path, "r|gz") as tf:   # streaming mode: one pass
        pbar = tqdm(desc="encoding", unit="utt")
        for member in tf:
            if limit and n_done >= limit:
                break
            name = Path(member.name)
            stem = name.stem.removesuffix(".normalized").removesuffix(".original")
            if name.suffix == ".wav":
                fh = tf.extractfile(member)
                data, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
                if sr != 24_000:
                    n_skipped += 1
                    continue
                speaker = name.parts[-3] if len(name.parts) >= 3 else "unknown"
                wavs[stem] = (data, speaker)
            elif name.name.endswith((".normalized.txt", ".original.txt")):
                if stem in texts:        # prefer .normalized over .original
                    continue
                texts[stem] = tf.extractfile(member).read().decode("utf-8").strip()
            else:
                continue
            if stem in wavs and stem in texts:
                emit(stem)
                pbar.update(1)
        pbar.close()
    flush()
    print(f"done: {n_done} encoded, {n_skipped} skipped, "
          f"{len(wavs)} wavs without transcript dropped")


def prepare_from_hf(out_dir: Path, shard_size: int = 4000,
                    limit: int | None = None, device: str = "cpu",
                    max_seconds: float = 13.0,
                    split: str = "train.clean.360",
                    prefix: str = "libritts_r",
                    concat_seconds: float = 0.0,
                    skip_utts: int = 0, shard_start: int = 0):
    """Stream LibriTTS-R from HuggingFace (mythicinfinity/libritts_r) and
    Mimi-encode on the fly — no archive download, no extraction.

    concat_seconds > 0: concatenate consecutive utterances from the SAME
    speaker+chapter (with a 0.3s gap) into segments up to that length, to
    teach the model to sustain long/multi-sentence speech. Item text is the
    joined sentences. Falls back to single utterances at chapter boundaries.
    """
    from datasets import Audio, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    config = "other" if "other" in split else "clean"
    ds = load_dataset("mythicinfinity/libritts_r", config, split=split,
                      streaming=True)
    # decode=False: skip datasets' torchcodec audio decoding (segfaults
    # without FFmpeg/torchcodec); we decode the flac/wav bytes via soundfile
    ds = ds.cast_column("audio", Audio(decode=False))
    # resume: skip utts already encoded in a prior run (HF streaming order is
    # deterministic, so .skip lands at the same point); shard_start avoids
    # overwriting existing shards on disk.
    if skip_utts:
        ds = ds.skip(skip_utts)
        print(f"resuming: skipped {skip_utts} utts, shards start at {shard_start}", flush=True)
    items, n_shard, n_done, n_skipped = [], shard_start, 0, 0
    gap = torch.zeros(int(0.3 * 24_000))  # inter-sentence silence

    def flush():
        nonlocal items, n_shard
        if items:
            path = out_dir / f"{prefix}_{n_shard:03d}.pt"
            torch.save(items, path)
            print(f"wrote {path.name} ({len(items)} items)", flush=True)
            items, n_shard = [], n_shard + 1

    def emit(wav, text, speaker, uid):
        nonlocal n_done
        items.append({"id": uid, "speaker": speaker, "text": text,
                      "codes": codec.encode(wav)})
        n_done += 1
        if len(items) >= shard_size:
            flush()

    # concatenation buffer (group = speaker+chapter)
    buf_wavs, buf_texts, buf_key, buf_id = [], [], None, None

    def buf_dur():
        return sum(len(w) for w in buf_wavs) / 24_000

    def flush_buf():
        nonlocal buf_wavs, buf_texts, buf_key, buf_id
        if buf_wavs:
            joined = buf_wavs[0]
            for w in buf_wavs[1:]:
                joined = torch.cat([joined, gap, w])
            emit(joined, " ".join(buf_texts), buf_key.split("/")[0], buf_id)
        buf_wavs, buf_texts, buf_key, buf_id = [], [], None, None

    for ex in tqdm(ds, desc="encoding", unit="utt"):
        if limit and n_done >= limit:
            break
        data, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if sr != 24_000 or len(data) > max_seconds * sr:
            n_skipped += 1
            continue
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        speaker = str(ex.get("speaker_id", "unknown"))
        text = (ex.get("text_normalized") or ex.get("text_original") or "").strip()
        uid = ex.get("id", str(n_done))

        if concat_seconds <= 0:
            emit(wav, text, speaker, uid)
            continue

        # group consecutive utts by speaker+chapter (id = spk_chapter_utt_seg)
        parts = uid.split("_")
        key = f"{speaker}/{parts[1] if len(parts) > 1 else '0'}"
        gap_dur = 0.3 if buf_wavs else 0.0
        if buf_key is not None and (key != buf_key or
                buf_dur() + gap_dur + len(wav) / 24_000 > concat_seconds):
            flush_buf()
        if buf_key is None:
            buf_key, buf_id = key, uid
        buf_wavs.append(wav)
        buf_texts.append(text)
    if concat_seconds > 0:
        flush_buf()
    flush()
    print(f"done: {n_done} encoded, {n_skipped} skipped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--from-tar", type=Path, default=None)
    p.add_argument("--from-hf", action="store_true",
                   help="stream mythicinfinity/libritts_r from HuggingFace")
    p.add_argument("--split", default="train.clean.360",
                   help='HF split, e.g. "train.other.500"')
    p.add_argument("--prefix", default="libritts_r",
                   help="shard filename prefix (use libritts_o for other-500)")
    p.add_argument("--concat-seconds", type=float, default=0.0,
                   help="concatenate same-speaker+chapter utts into segments "
                        "up to N seconds (teaches sustaining; e.g. 22)")
    p.add_argument("--skip-utts", type=int, default=0,
                   help="resume: skip first N utts (already encoded last run)")
    p.add_argument("--shard-start", type=int, default=0,
                   help="resume: number shards from here (avoid overwrite)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=4000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.from_hf:
        # raise per-utt cap when concatenating so long segments can form
        ms = max(a.concat_seconds, 13.0) if a.concat_seconds else 13.0
        prepare_from_hf(a.out, a.shard_size, a.limit, a.device,
                        max_seconds=ms, split=a.split, prefix=a.prefix,
                        concat_seconds=a.concat_seconds,
                        skip_utts=a.skip_utts, shard_start=a.shard_start)
    elif a.from_tar:
        prepare_from_tar(a.from_tar, a.out, a.shard_size, a.limit, a.device)
    elif a.root:
        prepare(a.root, a.out, a.shard_size, a.limit, a.device)
    else:
        raise SystemExit("give --root <dir>, --from-tar <tar.gz>, or --from-hf")
