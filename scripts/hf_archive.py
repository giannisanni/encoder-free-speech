"""Archive training checkpoints to a private HF repo, then free local disk.

Policy: every checkpoints/<run>/ckpt_*.pt is uploaded to
hf://<repo>/<run>/ckpt_<step>.pt (skipped if already there, size-checked).
After a verified upload, the local file is deleted UNLESS it is one of the
newest 2 in its run dir (kept for fast resume; finals stay by construction).
Files younger than 120 s are skipped (may still be mid-write).

Run via cron on substrate:
  */30 * * * * cd ~/gemma-csm && .venv/bin/python scripts/hf_archive.py >> ~/hf_archive.log 2>&1
"""
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "giannisan/gemma-csm-ckpts"
ROOT = Path.home() / "gemma-csm" / "checkpoints"
KEEP_LOCAL = 2

api = HfApi()
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
remote = {f.path: f.size for f in api.list_repo_tree(REPO, recursive=True)
          if f.path.endswith(".pt")}

for run_dir in sorted(ROOT.iterdir()):
    ckpts = sorted(run_dir.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for i, ck in enumerate(ckpts):
        if time.time() - ck.stat().st_mtime < 120:
            continue  # possibly mid-write
        dest = f"{run_dir.name}/{ck.name}"
        size = ck.stat().st_size
        if remote.get(dest) != size:
            print(f"upload {dest} ({size/1e6:.0f}MB)", flush=True)
            api.upload_file(path_or_fileobj=str(ck), path_in_repo=dest,
                            repo_id=REPO)
            remote[dest] = size
        if i < len(ckpts) - KEEP_LOCAL and remote.get(dest) == size:
            print(f"free  {ck} (archived)", flush=True)
            ck.unlink()
