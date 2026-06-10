import numpy as np
import soundfile as sf
import torch

from gemma_csm.data_prep_libritts import prepare


def test_prepare_libritts_structure(tmp_path):
    # mimic <root>/<speaker>/<chapter>/<utt>.wav + .normalized.txt
    root = tmp_path / "train-clean-360"
    for spk, utt in [("19", "19_198_000000_000000"), ("26", "26_495_000004_000000")]:
        d = root / spk / "198"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{utt}.wav"), np.zeros(24_000, dtype=np.float32), 24_000)
        (d / f"{utt}.normalized.txt").write_text("Hello there.")
    # one over-length clip that must be skipped
    long_dir = root / "19" / "199"
    long_dir.mkdir(parents=True)
    sf.write(str(long_dir / "long.wav"),
             np.zeros(24_000 * 20, dtype=np.float32), 24_000)
    (long_dir / "long.normalized.txt").write_text("Too long.")

    out = tmp_path / "shards"
    prepare(root, out, shard_size=10)
    shards = sorted(out.glob("libritts_r_*.pt"))
    assert len(shards) == 1
    items = torch.load(shards[0], weights_only=True)
    assert len(items) == 2                       # long clip skipped
    assert {it["speaker"] for it in items} == {"19", "26"}
    assert all(it["codes"].shape[0] == 32 for it in items)
