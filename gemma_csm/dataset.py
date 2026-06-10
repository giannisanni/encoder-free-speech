import torch
from torch.utils.data import Dataset


class CodesDataset(Dataset):
    """Items: {'text': str, 'codes': (32, T) long}. Tokenizes text lazily."""

    def __init__(self, shard_paths, tokenizer, max_frames: int = 160):
        self.items = []
        for p in shard_paths:
            self.items += [it for it in torch.load(p, weights_only=True)
                           if it["codes"].shape[1] <= max_frames]
        self.tok = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        # CSM convention: speaker id as a text prefix, so voice becomes
        # selectable at inference ("[1234] text..."). Items without a speaker
        # field (LJSpeech) get the stable id "lj".
        text = f"[{it.get('speaker', 'lj')}] {it['text']}"
        ids = self.tok(text, add_special_tokens=True)["input_ids"]
        return {"text_ids": torch.tensor(ids, dtype=torch.long), "codes": it["codes"]}


def collate(batch, text_pad_id: int):
    """Pad text left-aligned; pad codes to max frames + 1 (room for EOS target).

    frame_mask marks frames-with-content INCLUDING the EOS slot, so the c0
    loss learns to stop.
    """
    bsz = len(batch)
    tmax = max(len(b["text_ids"]) for b in batch)
    fmax = max(b["codes"].shape[1] for b in batch) + 1
    text_ids = torch.full((bsz, tmax), text_pad_id, dtype=torch.long)
    text_mask = torch.zeros(bsz, tmax, dtype=torch.bool)
    codes = torch.zeros(bsz, 32, fmax, dtype=torch.long)
    frame_mask = torch.zeros(bsz, fmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        t, f = len(b["text_ids"]), b["codes"].shape[1]
        text_ids[i, :t] = b["text_ids"]
        text_mask[i, :t] = True
        codes[i, :, :f] = b["codes"]
        frame_mask[i, : f + 1] = True
    return {"text_ids": text_ids, "text_mask": text_mask,
            "codes": codes, "frame_mask": frame_mask}
