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
import tarfile
from pathlib import Path

import soundfile as sf
import torch

from gemma_csm.codec import MimiCodec

REPO = "CASIA-LM/OpenS2S_Datasets"
QUERY_TAR = f"datasets/{REPO}/en_query_wav.tar.gz"
RESP_TAR0 = f"datasets/{REPO}/en_response_wav.tar.gz.0"


def _mono(data):
    return data.mean(-1) if data.ndim > 1 else data


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
                   max_seconds: float = 20.0, skip: int = 0, shard_start: int = 0):
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

    # first-K encountered from each scattered tar (leading bytes), intersect
    k = max(int(n_pairs * 1.3), 600)
    fs = HfFileSystem()
    print(f"streaming first ~{k} query wavs ...", flush=True)
    qw = _stream_first(fs, QUERY_TAR, k, skip)
    print(f"  {len(qw)} query wavs; streaming response ...", flush=True)
    rw = _stream_first(fs, RESP_TAR0, k, skip)
    pairs = [i for i in qw if i in rw and i in meta]
    print(f"  {len(rw)} response wavs; {len(pairs)} paired indices", flush=True)

    items, n_shard, n_done, n_skip = [], shard_start, 0, 0

    def flush():
        nonlocal items, n_shard
        if items:
            torch.save(items, out_dir / f"opens2s_en_{n_shard:03d}.pt")
            print(f"wrote opens2s_en_{n_shard:03d}.pt ({len(items)})", flush=True)
            items, n_shard = [], n_shard + 1

    for idx in pairs[:n_pairs]:
        d = meta[idx]
        rdata, rsr = sf.read(io.BytesIO(rw[idx]), dtype="float32")
        rdata = _mono(rdata)
        if len(rdata) > max_seconds * rsr:
            n_skip += 1
            continue
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
    print(f"done: {n_done} encoded, {n_skip} skipped", flush=True)


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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.stream:
        prepare_stream(a.out, a.n_pairs, a.shard_size, a.device, a.backbone_id,
                       skip=a.skip, shard_start=a.shard_start)
    else:
        raise SystemExit("only --stream mode is supported (EN wavs are ~88GB)")
