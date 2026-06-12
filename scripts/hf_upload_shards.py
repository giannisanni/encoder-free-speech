"""Mirror Mimi-code dataset shards to a private HF dataset repo.

Upload-only (never deletes local files — training needs them on disk).
Covers every data/shards*/ directory, so new corpora (other-500, Emilia,
val sets) are picked up automatically. Files younger than 120 s are
skipped (possibly mid-write by an encoder).

Shards derive from LibriTTS-R (CC BY 4.0) and LJSpeech (public domain);
repo kept private regardless.

Cron (substrate):
  */30 * * * * cd ~/gemma-csm && flock -n /tmp/hf_shards.lock .venv/bin/python scripts/hf_upload_shards.py >> ~/hf_shards.log 2>&1
"""
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "giannisan/gemma-csm-shards"
ROOT = Path.home() / "gemma-csm" / "data"

api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
remote = {f.path: f.size
          for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
          if f.path.endswith(".pt")}

for shard_dir in sorted(ROOT.glob("shards*")):
    for pt in sorted(shard_dir.glob("*.pt")):
        if time.time() - pt.stat().st_mtime < 120:
            continue
        dest = f"{shard_dir.name}/{pt.name}"
        size = pt.stat().st_size
        if remote.get(dest) != size:
            print(f"upload {dest} ({size/1e6:.0f}MB)", flush=True)
            api.upload_file(path_or_fileobj=str(pt), path_in_repo=dest,
                            repo_id=REPO, repo_type="dataset")
            remote[dest] = size
