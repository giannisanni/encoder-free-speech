"""Encode VoiceAssistant-400K (general voice-assistant Q&A) to unified S2S shards,
to MIX with the empathy-only OpenS2S data so the unified model generalizes
(answers factual / identity / general queries instead of collapsing to OpenS2S
empathy templates).

Source: worstchan/VoiceAssistant-400K-SLAM-Omni (HF). It has REAL spoken
`question_audio` + `answer` text, but its answer audio is CosyVoice tokens (not
Mimi, unusable here). So we take the query audio + answer text and SYNTHESIZE the
reply audio ourselves with Kokoro TTS (one voice), then Mimi-encode it.

Per pair we emit the SAME schema as data_prep_opens2s (drop-in for OpenS2SDataset):
  input_features : question_audio -> Gemma processor
  codes          : Kokoro(answer text) -> Mimi RVQ codes
  text           : answer ;  query_text : question

Usage (on a GPU pod with kokoro + datasets installed):
  python -m gemma_csm.data_prep_voiceassistant --n-pairs 15000 \\
      --out data/shards_va --voice af_heart
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from gemma_csm.codec import MimiCodec

REPO = "worstchan/VoiceAssistant-400K-SLAM-Omni"


def _mono(a):
    a = np.asarray(a, dtype="float32")
    return a.mean(-1) if a.ndim > 1 else a


def prepare(out_dir: Path, n_pairs: int, shard_size: int, device: str,
            backbone_id: str, voice: str, max_seconds: float, max_words: int,
            min_words: int, split: str):
    from datasets import load_dataset
    from transformers import AutoProcessor
    from kokoro import KPipeline

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(backbone_id)
    codec = MimiCodec(device=device)
    tts = KPipeline(lang_code="a")   # American English
    SR = 24000                       # Kokoro and Mimi are both 24 kHz

    print(f"streaming {REPO} [{split}] ...", flush=True)
    ds = load_dataset(REPO, split=split, streaming=True)

    items, n_shard, n_done, n_skip = [], 0, 0, 0

    def flush():
        nonlocal items, n_shard
        if items:
            dst = out_dir / f"va_en_{n_shard:03d}.pt"
            tmp = out_dir / f"va_en_{n_shard:03d}.pt.tmp"
            torch.save(items, tmp); os.replace(tmp, dst)
            print(f"wrote {dst.name} ({len(items)})", flush=True)
            items, n_shard = [], n_shard + 1

    def synth(text):
        chunks = [a for _g, _p, a in tts(text, voice=voice)]
        if not chunks:
            return None
        wav = torch.cat([c if torch.is_tensor(c) else torch.from_numpy(np.asarray(c))
                         for c in chunks]).float()
        if wav.numel() > max_seconds * SR:
            wav = wav[: int(max_seconds * SR)]
        return wav

    for ex in ds:
        if n_done >= n_pairs:
            break
        ans = (ex.get("answer") or "").strip()
        w = len(ans.split())
        if not ans or w < min_words:
            n_skip += 1
            continue
        if max_words and w > max_words:
            ans = " ".join(ans.split()[:max_words])
        qa = ex.get("question_audio")
        if not qa or "array" not in qa:
            n_skip += 1
            continue
        try:
            qwav = _mono(qa["array"]); qsr = int(qa.get("sampling_rate", 16000))
            rwav = synth(ans)
            if rwav is None or rwav.numel() < SR * 0.2:
                n_skip += 1
                continue
            ev = codec.encode.__code__.co_varnames
            codes = (codec.encode(rwav, sample_rate=SR) if "sample_rate" in ev
                     else codec.encode(rwav)).detach().to("cpu")
            feat = proc(audio=qwav, sampling_rate=qsr, return_tensors="pt")
            inf = None
            for k in ("input_features", "audio_values", "input_audio_embeds"):
                if k in feat:
                    inf = feat[k]; break
            if inf is None:
                raise KeyError(f"no audio features; keys {list(feat.keys())}")
        except Exception as e:
            print(f"  skip {n_done}: {type(e).__name__}: {e}", flush=True)
            n_skip += 1
            continue
        items.append({
            "index": int(ex.get("index", n_done)),
            "input_features": inf.squeeze(0).to(torch.float16),
            "codes": codes,
            "text": ans,
            "query_text": (ex.get("question") or "").strip(),
            "emotion": "neutral",
        })
        n_done += 1
        if n_done % 200 == 0:
            print(f"  encoded {n_done}/{n_pairs} (skipped {n_skip})", flush=True)
        if len(items) >= shard_size:
            flush()
    flush()
    print(f"VA_DONE: {n_done} encoded, {n_skip} skipped", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-pairs", type=int, default=15000)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone-id", default="google/gemma-4-12B-it")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--max-words", type=int, default=70, help="truncate long answers")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--split", default="train")
    a = p.parse_args()
    prepare(a.out, a.n_pairs, a.shard_size, a.device, a.backbone_id, a.voice,
            a.max_seconds, a.max_words, a.min_words, a.split)
