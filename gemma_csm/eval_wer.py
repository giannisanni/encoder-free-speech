"""WER harness: transcribe generated clips with Whisper, score against references.

The objective stand-in for blind human listening (rev. 5 spec): Whisper has
never seen our prompts, so its transcription is a tireless blind listener.
Normalization applied to BOTH sides per standard practice.

  python -m gemma_csm.eval_wer --manifest manifest.tsv [--model small]

Manifest: tab-separated lines of  <wav_path> <tab> <reference text>
Outputs per-clip WER + what Whisper heard, sorted by manifest order.
"""
import argparse
import csv
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer

NORM = EnglishTextNormalizer()


def transcribe(model: WhisperModel, wav: Path) -> str:
    segments, _info = model.transcribe(str(wav), language="en", beam_size=5)
    return " ".join(s.text for s in segments).strip()


def word_recall(ref_n: str, hyp_n: str) -> float:
    """Fraction of reference words present in the hypothesis (multiset).

    Bounded 0-1 and immune to hallucinated insertions, unlike WER, which
    saturates >100% when Whisper confabulates fluent text from mush. This is
    the machine analogue of a human listener reporting 'I caught N words'.
    """
    from collections import Counter
    ref_c, hyp_c = Counter(ref_n.split()), Counter(hyp_n.split())
    if not ref_c:
        return float("nan")
    hits = sum(min(c, hyp_c[w]) for w, c in ref_c.items())
    return hits / sum(ref_c.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", default="small", help="faster-whisper size")
    a = p.parse_args()

    model = WhisperModel(a.model, device="cpu", compute_type="int8")
    rows = []
    with open(a.manifest) as fh:
        for wav_path, ref in csv.reader(fh, delimiter="\t"):
            if not Path(wav_path).exists():
                print(f"  skip  {wav_path} (missing)")
                continue
            hyp = transcribe(model, Path(wav_path))
            ref_n, hyp_n = NORM(ref), NORM(hyp)
            score = wer(ref_n, hyp_n) if ref_n else float("nan")
            recall = word_recall(ref_n, hyp_n)
            rows.append((Path(wav_path).name, score, recall, hyp))
            print(f"WER {score:5.0%}  recall {recall:4.0%}  "
                  f"{Path(wav_path).name:42}  heard: {hyp!r}")
    valid = [(r[1], r[2]) for r in rows if r[1] == r[1]]
    if valid:
        print(f"\n{len(valid)} clips:  mean WER {sum(v[0] for v in valid)/len(valid):.0%}"
              f"   mean recall {sum(v[1] for v in valid)/len(valid):.0%}")


if __name__ == "__main__":
    main()
