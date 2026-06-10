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


def test_prepare_from_tar_streaming(tmp_path):
    import tarfile
    from gemma_csm.data_prep_libritts import prepare_from_tar

    # build a tiny LibriTTS-shaped tar.gz
    src = tmp_path / "train-clean-360" / "19" / "198"
    src.mkdir(parents=True)
    sf.write(str(src / "19_198_0.wav"), np.zeros(24_000, dtype=np.float32), 24_000)
    (src / "19_198_0.normalized.txt").write_text("Hello.")
    tar_path = tmp_path / "tc360.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tmp_path / "train-clean-360", arcname="train-clean-360")

    out = tmp_path / "shards"
    prepare_from_tar(tar_path, out)
    items = torch.load(sorted(out.glob("*.pt"))[0], weights_only=True)
    assert len(items) == 1
    assert items[0]["speaker"] == "19"
    assert items[0]["text"] == "Hello."
    assert items[0]["codes"].shape[0] == 32
