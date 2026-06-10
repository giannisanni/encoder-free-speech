import torch

from gemma_csm.dataset import CodesDataset, collate


class FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [101] * len(text.split())}


def _shard(tmp_path):
    items = [
        {"id": "a", "text": "hello there", "codes": torch.zeros(32, 20, dtype=torch.long)},
        {"id": "b", "text": "one two three four", "codes": torch.ones(32, 35, dtype=torch.long)},
    ]
    p = tmp_path / "s.pt"; torch.save(items, p); return p


def test_dataset_and_collate(tmp_path):
    ds = CodesDataset([_shard(tmp_path)], FakeTok(), max_frames=160)
    assert len(ds) == 2
    batch = collate([ds[0], ds[1]], text_pad_id=0)
    assert batch["text_ids"].shape[0] == 2
    assert batch["codes"].shape == (2, 32, 36)        # padded to max T+1 (EOS frame)
    assert batch["frame_mask"].shape == (2, 36)
    assert batch["frame_mask"][0].sum() == 21          # 20 frames + 1 EOS slot


def test_speaker_prefix(tmp_path):
    items = [
        {"id": "a", "speaker": "1234", "text": "hello", "codes": torch.zeros(32, 5, dtype=torch.long)},
        {"id": "b", "text": "hello", "codes": torch.zeros(32, 5, dtype=torch.long)},  # no speaker -> lj
    ]
    p = tmp_path / "s.pt"; torch.save(items, p)

    captured = []
    class Tok:
        def __call__(self, text, **kw):
            captured.append(text)
            return {"input_ids": [1, 2]}

    ds = CodesDataset([p], Tok())
    ds[0]; ds[1]
    assert captured == ["[1234] hello", "[lj] hello"]
