"""Encode OpenS2S (CASIA-LM/OpenS2S_Datasets) English speech-to-speech pairs to
training shards for the audio-IN (RQ1) run.

manifest_en.jsonl schema (verified):
  {index, query:{audio,text,emotion}, response:{audio,text,emotion}}
  query.audio  = "en_query_wav/<i>.wav"     (INPUT speech, prosody to condition on)
  response.audio = "en_response_wav/<i>.wav" (OUTPUT speech, the target) + text to say

Per pair we emit:
  input_features : query speech -> Gemma processor (fed to get_audio_features
                   at train time -> audio_in_embeds, the audio-IN segment)
  codes          : response speech -> Mimi RVQ codes (speech-out target)
  text           : response.text ; query_text / emotion kept for analysis

The EN wavs are ~88 GB compressed (query 19 GB + response 69 GB), so we do NOT
download/extract them. --stream opens the tar.gz over an HF filesystem stream
and takes the first --n-pairs by index, encoding on the fly (only output shards
hit disk). The split response tar streams from its .0 part (the head of the
gzip), which holds the early indices.

REQUIRES the audio-IN venv: transformers-from-source (>=5.13, Gemma4UnifiedProcessor)
+ torchvision + pillow + soundfile + Mimi codec.

Usage (stream a subset):
  python -m gemma_csm.data_prep_opens2s --stream --n-pairs 2000 \\
      --out data/shards_opens2s
"""
import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import soundfile as sf
import torch

from gemma_csm.codec import MimiCodec

REPO = "CASIA-LM/OpenS2S_Datasets"
QUERY_TAR = f"datasets/{REPO}/en_query_wav.tar.gz"
RESP_TAR0 = f"datasets/{REPO}/en_response_wav.tar.gz.0"
RESP_TARS = [f"datasets/{REPO}/en_response_wav.tar.gz.{i}" for i in range(7)]


def _mono(data):
    return data.mean(-1) if data.ndim > 1 else data


class _ChainedFile:
    """Read a list of fsspec paths as one continuous byte stream (a tar.gz split
    with `split` into .0,.1,... must be concatenated to decompress past .0)."""

    def __init__(self, fs, paths):
        self._fs = fs
        self._paths = list(paths)
        self._i = 0
        self._fh = fs.open(self._paths[0], "rb")

    def _advance(self):
        if self._i < len(self._paths) - 1:
            self._i += 1
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = self._fs.open(self._paths[self._i], "rb")
            return True
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            chunks = [self._fh.read()]
            while self._advance():
                chunks.append(self._fh.read())
            return b"".join(chunks)
        out = bytearray()
        while len(out) < size:
            chunk = self._fh.read(size - len(out))
            if chunk:
                out += chunk
            elif not self._advance():
                break
        return bytes(out)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _stream_first(fs, path, k, skip=0):
    """{index: wav_bytes} for the first k .wav members in a streamed tar.gz.
    OpenS2S tars are not index-ordered, but the same indices cluster at the
    start of both query and response tars, so first-k-of-each + intersect
    yields paired indices from leading bytes only (no full 88GB download)."""
    got = {}
    seen = 0
    f = fs.open(path, "rb")
    tf = tarfile.open(fileobj=f, mode="r|gz")
    for m in tf:
        if not m.isfile() or not m.name.endswith(".wav"):
            continue
        try:
            idx = int(Path(m.name).stem)
        except ValueError:
            continue
        seen += 1
        if seen <= skip:
            continue
        got[idx] = tf.extractfile(m).read()
        if len(got) >= k:
            break
    return got


def prepare_stream(out_dir: Path, n_pairs: int, shard_size: int = 1000,
                   device: str = "cpu", backbone_id: str = "google/gemma-4-12B-it",
                   max_seconds: float = 20.0, skip: int = 0, shard_start: int = 0,
                   min_words: int = 0, candidates: int = 0):
    from huggingface_hub import HfFileSystem, hf_hub_download
    from transformers import AutoProcessor

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(backbone_id)
    codec = MimiCodec(device=device)

    # full manifest (index -> texts/emotion); ~50k lines, fits in RAM
    mpath = hf_hub_download(REPO, "manifest_en.jsonl", repo_type="dataset")
    meta = {}
    with open(mpath) as f:
        for line in f:
            d = json.loads(line)
            meta[int(d["index"])] = d
    print(f"manifest: {len(meta)} pairs total", flush=True)

    # first-K encountered from each scattered tar (leading bytes), intersect.
    # when filtering by --min-words most candidates are discarded, so stream a
    # bigger pool (--candidates) than the target n_pairs.
    k = candidates if candidates > 0 else max(int(n_pairs * 1.3), 600)
    fs = HfFileSystem()
    print(f"streaming first ~{k} query wavs ...", flush=True)
    qw = _stream_first(fs, QUERY_TAR, k, skip)
    print(f"  {len(qw)} query wavs; streaming response ...", flush=True)
    rw = _stream_first(fs, RESP_TAR0, k, skip)
    pairs = [i for i in qw if i in rw and i in meta]
    print(f"  {len(rw)} response wavs; {len(pairs)} paired indices", flush=True)

    items, n_shard, n_done, n_skip, n_short = [], shard_start, 0, 0, 0

    def flush():
        nonlocal items, n_shard
        if items:
            torch.save(items, out_dir / f"opens2s_en_{n_shard:03d}.pt")
            print(f"wrote opens2s_en_{n_shard:03d}.pt ({len(items)})", flush=True)
            items, n_shard = [], n_shard + 1

    for idx in pairs:
        if n_done >= n_pairs:
            break
        d = meta[idx]
        if min_words and len((d["response"].get("text", "") or "").split()) < min_words:
            n_short += 1
            continue
        rdata, rsr = sf.read(io.BytesIO(rw[idx]), dtype="float32")
        rdata = _mono(rdata)
        if len(rdata) > max_seconds * rsr:
            rdata = rdata[: int(max_seconds * rsr)]  # truncate over-long; keep it (long/short mix)
            n_skip += 1
        ev = codec.encode.__code__.co_varnames
        codes = (codec.encode(torch.from_numpy(rdata), sample_rate=rsr)
                 if "sample_rate" in ev else codec.encode(torch.from_numpy(rdata)))
        qdata, qsr = sf.read(io.BytesIO(qw[idx]), dtype="float32")
        feat = proc(audio=_mono(qdata), sampling_rate=qsr, return_tensors="pt")
        input_features = None
        for k in ("input_features", "audio_values", "input_audio_embeds"):
            if k in feat:
                input_features = feat[k]
                break
        if input_features is None:
            raise KeyError(f"no audio features; processor keys: {list(feat.keys())}")
        items.append({
            "index": idx,
            "input_features": input_features.squeeze(0).to(torch.float16),
            "codes": codes,
            "text": d["response"].get("text", ""),
            "query_text": d["query"].get("text", ""),
            "emotion": d["query"].get("emotion") or d["response"].get("emotion"),
        })
        n_done += 1
        if len(items) >= shard_size:
            flush()
    flush()
    print(f"done: {n_done} encoded, {n_short} skipped (<{min_words} words), "
          f"{n_skip} truncated (capped at {max_seconds}s)", flush=True)


def prepare_stream_full(out_dir: Path, target: int, shard_size: int = 1000,
                        device: str = "cpu", backbone_id: str = "google/gemma-4-12B-it",
                        max_seconds: float = 20.0, shard_start: int = 0):
    """Add `target` NEW pairs disjoint from existing shards, reading ALL response
    pieces. Single pass, bounded RAM (holds ~target*1.5 response + target query)."""
    import glob as _glob
    from huggingface_hub import HfFileSystem, hf_hub_download
    from transformers import AutoProcessor

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for sf_path in _glob.glob(str(out_dir / "opens2s_en_*.pt")):
        try:
            for it in torch.load(sf_path, weights_only=True):
                seen.add(int(it["index"]))
        except Exception:
            pass
    print(f"already have {len(seen)} pairs; targeting +{target} new", flush=True)

    mpath = hf_hub_download(REPO, "manifest_en.jsonl", repo_type="dataset")
    meta = {}
    with open(mpath) as fh:
        for line in fh:
            d = json.loads(line); meta[int(d["index"])] = d
    print(f"manifest: {len(meta)} pairs total", flush=True)

    fs = HfFileSystem()
    cand = int(target * 1.5)
    print(f"streaming response (all {len(RESP_TARS)} pieces) for up to {cand} new ...", flush=True)
    rw = {}
    ch = _ChainedFile(fs, RESP_TARS)
    tf = tarfile.open(fileobj=ch, mode="r|gz")
    for m in tf:
        if not m.isfile() or not m.name.endswith(".wav"):
            continue
        try:
            idx = int(Path(m.name).stem)
        except ValueError:
            continue
        if idx in seen or idx not in meta:
            continue
        rw[idx] = tf.extractfile(m).read()
        if len(rw) % 1000 == 0:
            print(f"  ...{len(rw)} new response wavs", flush=True)
        if len(rw) >= cand:
            break
    tf.close(); ch.close()
    print(f"  collected {len(rw)} new response wavs", flush=True)

    want = set(rw); qw = {}
    print("streaming query for those indices ...", flush=True)
    fs = HfFileSystem()  # fresh client: pass-1 handle closes kill the shared one
    qf = fs.open(QUERY_TAR, "rb")
    qtf = tarfile.open(fileobj=qf, mode="r|gz")
    for m in qtf:
        if not m.isfile() or not m.name.endswith(".wav"):
            continue
        try:
            idx = int(Path(m.name).stem)
        except ValueError:
            continue
        if idx in want:
            try:
                qw[idx] = qtf.extractfile(m).read()
            except Exception as e:
                print(f"  query read skip {idx}: {type(e).__name__}", flush=True)
                continue
            if len(qw) % 1000 == 0:
                print(f"  ...{len(qw)} query matched", flush=True)
            if len(qw) >= len(want):
                break
    qtf.close(); qf.close()
    print(f"  matched {len(qw)} query wavs", flush=True)

    proc = AutoProcessor.from_pretrained(backbone_id)
    codec = MimiCodec(device=device)
    items, n_shard, n_done, n_skip = [], shard_start, 0, 0

    def flush():
        nonlocal items, n_shard
        if items:
            dst = out_dir / f"opens2s_en_{n_shard:03d}.pt"
            tmp = out_dir / f"opens2s_en_{n_shard:03d}.pt.tmp"
            torch.save(items, tmp); os.replace(tmp, dst)
            print(f"wrote {dst.name} ({len(items)})", flush=True)
            items, n_shard = [], n_shard + 1

    for idx in list(qw):
        if n_done >= target:
            break
        d = meta[idx]
        rdata, rsr = sf.read(io.BytesIO(rw[idx]), dtype="float32")
        rdata = _mono(rdata)
        if len(rdata) > max_seconds * rsr:
            rdata = rdata[: int(max_seconds * rsr)]  # truncate over-long; keep it (long/short mix)
            n_skip += 1
        ev = codec.encode.__code__.co_varnames
        codes = (codec.encode(torch.from_numpy(rdata), sample_rate=rsr)
                 if "sample_rate" in ev else codec.encode(torch.from_numpy(rdata)))
        qdata, qsr = sf.read(io.BytesIO(qw[idx]), dtype="float32")
        feat = proc(audio=_mono(qdata), sampling_rate=qsr, return_tensors="pt")
        input_features = None
        for kk in ("input_features", "audio_values", "input_audio_embeds"):
            if kk in feat:
                input_features = feat[kk]
                break
        if input_features is None:
            raise KeyError(f"no audio features; keys: {list(feat.keys())}")
        items.append({
            "index": idx,
            "input_features": input_features.squeeze(0).to(torch.float16),
            "codes": codes,
            "text": d["response"].get("text", ""),
            "query_text": d["query"].get("text", ""),
            "emotion": d["query"].get("emotion") or d["response"].get("emotion"),
        })
        n_done += 1
        if len(items) >= shard_size:
            flush()
    flush()
    print(f"done: {n_done} encoded, {n_skip} truncated (over-long capped at {max_seconds}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stream", action="store_true",
                   help="stream a subset from HF tars (no full download)")
    p.add_argument("--n-pairs", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--backbone-id", default="google/gemma-4-12B-it")
    p.add_argument("--skip", type=int, default=0, help="skip first N wavs (grow disjointly)")
    p.add_argument("--shard-start", type=int, default=0, help="start shard numbering here")
    p.add_argument("--full", action="store_true",
                   help="read ALL response pieces, dedupe vs existing shards, add --target new")
    p.add_argument("--target", type=int, default=5000, help="new pairs to add in --full mode")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="cap response length; over-long clips are TRUNCATED (kept), not skipped")
    p.add_argument("--min-words", type=int, default=0,
                   help="only keep response texts with >= N words (long-sentence dataset)")
    p.add_argument("--candidates", type=int, default=0,
                   help="stream N candidate wavs to filter from (default ~1.3x n-pairs); "
                        "set high when --min-words discards most candidates")
    a = p.parse_args()
    if a.full:
        prepare_stream_full(a.out, a.target, a.shard_size, a.device, a.backbone_id,
                            max_seconds=a.max_seconds, shard_start=a.shard_start)
    elif a.stream:
        prepare_stream(a.out, a.n_pairs, a.shard_size, a.device, a.backbone_id,
                       max_seconds=a.max_seconds, skip=a.skip, shard_start=a.shard_start,
                       min_words=a.min_words, candidates=a.candidates)
    else:
        raise SystemExit("use --full (recommended) or --stream")
