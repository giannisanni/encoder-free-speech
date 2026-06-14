"""OpenS2S audio-IN dataset + collate (paper rev. 27/28).

Items come from data_prep_opens2s shards:
  {index, input_features (Ta,640), codes (32,T), text, query_text, emotion}

collate pads text / codes / input_features and builds masks. The training loop
then runs audio_encode(input_features, input_features_mask) -> audio_in_embeds
(time dim preserved 1:1 by the audio tower), so audio_in_mask = the
input_features padding mask. The padded batch feeds GemmaCSM.forward's
[text | audio-in | audio-out] path.
"""
import glob

import torch
from torch.utils.data import Dataset


class OpenS2SDataset(Dataset):
    def __init__(self, shards):
        if isinstance(shards, str):
            shards = glob.glob(shards)
        self.items = []
        for p in sorted(shards):
            self.items.extend(torch.load(p, weights_only=True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_opens2s(batch, tok, max_audio=None, max_frames=None, max_text=None):
    bsz = len(batch)
    text_tok = [tok(b["text"])["input_ids"][:max_text] for b in batch]
    tmax = max(len(t) for t in text_tok)
    fmax = max(min(b["codes"].shape[1], max_frames or 10**9) for b in batch) + 1
    amax = max(min(b["input_features"].shape[0], max_audio or 10**9) for b in batch)
    aed = batch[0]["input_features"].shape[1]
    pad = tok.pad_token_id or 0

    text_ids = torch.full((bsz, tmax), pad, dtype=torch.long)
    text_mask = torch.zeros(bsz, tmax, dtype=torch.bool)
    codes = torch.zeros(bsz, 32, fmax, dtype=torch.long)
    frame_mask = torch.zeros(bsz, fmax, dtype=torch.bool)
    input_features = torch.zeros(bsz, amax, aed, dtype=torch.float32)
    input_features_mask = torch.zeros(bsz, amax, dtype=torch.bool)

    for i, b in enumerate(batch):
        t = text_tok[i]
        cd = b["codes"][:, :max_frames] if max_frames else b["codes"]
        af = b["input_features"][:max_audio] if max_audio else b["input_features"]
        f = cd.shape[1]
        a = af.shape[0]
        text_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        text_mask[i, :len(t)] = True
        codes[i, :, :f] = cd
        frame_mask[i, :f + 1] = True
        input_features[i, :a] = af.float()
        input_features_mask[i, :a] = True

    return {
        "text_ids": text_ids, "text_mask": text_mask,
        "codes": codes, "frame_mask": frame_mask,
        "input_features": input_features,
        "input_features_mask": input_features_mask,
    }
