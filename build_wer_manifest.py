"""Reproduce the exact 5 unseen prompts assess_decode used, and write a
wav<TAB>reference manifest for eval_wer."""
import glob
from gemma_csm.dataset_opens2s import OpenS2SDataset

items = []
for f in sorted(glob.glob("data/shards_opens2s/opens2s_en_00[2-9].pt")):
    items += OpenS2SDataset([f]).items
items = items[:40]
prompts = [it.get("text", "").strip() for it in items]
prompts = [t for t in prompts if 20 <= len(t) <= 140][:5]

with open("wer_manifest.tsv", "w") as w:
    for i, t in enumerate(prompts):
        ref = t.replace("\t", " ").replace("\n", " ")
        w.write(f"checkpoints/a1_speak/samples/step014000_s{i}.wav\t{ref}\n")

print(f"manifest written: {len(prompts)} clips")
for i, t in enumerate(prompts):
    print(f"  s{i}: {t!r}")
