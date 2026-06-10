import numpy as np
import soundfile as sf
import torch

from gemma_csm.data_prep import prepare_shard


def test_prepare_shard(tmp_path):
    wav_dir = tmp_path / "wavs"; wav_dir.mkdir()
    for name in ("LJ001-0001", "LJ001-0002"):
        sf.write(str(wav_dir / f"{name}.wav"),
                 np.zeros(22050, dtype=np.float32), 22050)  # 1 s silence @ LJ rate
    meta = tmp_path / "metadata.csv"
    with open(meta, "w") as f:
        f.write("LJ001-0001|raw|Printing, then.\nLJ001-0002|raw|A second clip.\n")

    out = prepare_shard(meta, wav_dir, tmp_path / "out", limit=None)
    items = torch.load(out, weights_only=True)
    assert len(items) == 2
    assert items[0]["codes"].shape[0] == 32
    assert isinstance(items[0]["text"], str) and items[0]["text"] == "Printing, then."
