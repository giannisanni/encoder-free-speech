# Gemma-CSM Milestone-0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove a Gemma 4 e2b backbone can generate intelligible speech (Mimi audio codes) — M0a: overfit 50 LJSpeech clips; M0b: full LJSpeech single-voice TTS.

**Architecture:** Hierarchical CSM-style codec LM. Gemma 4 e2b (4-bit NF4 + LoRA) predicts Mimi codebook-0 per 12.5 Hz frame via a new linear head; a small from-scratch depth decoder predicts codebooks 1–31 from the backbone hidden state. Frozen Mimi encodes targets / decodes generations. See `docs/plans/2026-06-10-gemma-csm-design.md` for rationale.

**Tech Stack:** PyTorch + `transformers>=5.10.1` (Gemma4 + Mimi classes) + `peft` + `bitsandbytes`, `torchaudio`, `soundfile`. Python 3.12 venv on substrate (RTX 4060 Ti 16 GB, CUDA 12). Dev on Mac in `~/workspace/gemma-csm`, synced to substrate via rsync; training runs on substrate.

**Conventions:**
- All paths relative to repo root `~/workspace/gemma-csm` (both machines).
- `pytest` for tests; CPU-only tests must pass on the Mac, GPU tests are marked `@pytest.mark.gpu` and run on substrate.
- Commit after every green test. Small commits.

---

## Task 0: User actions (BLOCKING — cannot be done by the engineer)

1. Accept the Gemma license once at https://huggingface.co/google/gemma-4-E2B-it (click "Agree").
2. On substrate: `hf auth login` with a HuggingFace token that has read access (or confirm `~/.cache/huggingface/token` already exists).

Everything below assumes these are done.

---

## Task 1: Repo skeleton

**Files:**
- Create: `.gitignore`, `README.md`, `requirements.txt`, `gemma_csm/__init__.py`, `gemma_csm/config.py`, `tests/__init__.py`, `pytest.ini`

**Step 1: Write files**

`.gitignore`:
```
__pycache__/
*.pt
*.wav
data/
checkpoints/
samples/
.venv/
.pytest_cache/
```

`requirements.txt`:
```
torch>=2.4
torchaudio>=2.4
transformers>=5.10.1
peft>=0.13
bitsandbytes>=0.44
accelerate>=1.0
soundfile
numpy
pytest
tqdm
```

`pytest.ini`:
```ini
[pytest]
markers =
    gpu: requires CUDA GPU (run on substrate only)
addopts = -m "not gpu"
```

`gemma_csm/config.py`:
```python
from dataclasses import dataclass


@dataclass
class GemmaCSMConfig:
    backbone_id: str = "google/gemma-4-E2B-it"
    mimi_id: str = "kyutai/mimi"

    # Mimi codec geometry
    n_codebooks: int = 32          # set 8 for fast iteration (codes are precomputed for all 32)
    codebook_size: int = 2048
    frame_rate: float = 12.5
    sample_rate: int = 24_000

    # c0 head: codebook vocab + 1 EOS class
    c0_eos_id: int = 2048

    # depth decoder
    depth_d_model: int = 1024
    depth_layers: int = 4
    depth_heads: int = 8
    depth_loss_frame_frac: float = 1 / 16  # CSM memory trick

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # training
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    max_frames: int = 160          # ~12.8 s clips max for M0
```

`README.md`: two paragraphs — what the project is, link to both docs in `docs/plans/`. Keep short.

**Step 2: Verify pytest runs clean**

Run: `cd ~/workspace/gemma-csm && python3 -m venv .venv && .venv/bin/pip -q install pytest && .venv/bin/pytest`
Expected: `no tests ran` (exit 5 is fine).

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: repo skeleton and config"
```

---

## Task 2: Substrate environment + sync script

**Files:**
- Create: `scripts/sync-to-substrate.sh`, `scripts/setup-substrate.sh`

**Step 1: Write sync script**

`scripts/sync-to-substrate.sh`:
```bash
#!/usr/bin/env bash
# Push working tree (not data/checkpoints) Mac -> substrate.
set -euo pipefail
rsync -av --delete \
  --exclude .git --exclude .venv --exclude data --exclude checkpoints \
  --exclude samples --exclude __pycache__ \
  ~/workspace/gemma-csm/ gianni@substrate:~/gemma-csm/
```

`scripts/setup-substrate.sh` (run ON substrate):
```bash
#!/usr/bin/env bash
set -euo pipefail
cd ~/gemma-csm
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

**Step 2: Run both**

Run: `chmod +x scripts/*.sh && scripts/sync-to-substrate.sh && ssh gianni@substrate 'bash ~/gemma-csm/scripts/setup-substrate.sh'`
Expected: ends with `cuda: True`. (~5–10 min install; torch wheel is large.)

**Step 3: Verify gated model access**

Run: `ssh gianni@substrate '~/gemma-csm/.venv/bin/python -c "from huggingface_hub import auth_check; auth_check(\"google/gemma-4-E2B-it\"); print(\"access OK\")"'`
Expected: `access OK`. If `GatedRepoError`: Task 0 wasn't completed — stop and tell the user.

**Step 4: Commit**

```bash
git add scripts && git commit -m "feat: substrate sync and setup scripts"
```

---

## Task 3: Mimi codec wrapper (encode/decode roundtrip)

The whole project rests on Mimi behaving as expected (32 codebooks @ 12.5 Hz @ 24 kHz). Verify that FIRST, with a real listenable roundtrip.

**Files:**
- Create: `gemma_csm/codec.py`, `tests/test_codec.py`

**Step 1: Write the failing test**

`tests/test_codec.py`:
```python
import pytest
import torch

from gemma_csm.codec import MimiCodec


@pytest.fixture(scope="module")
def codec():
    return MimiCodec(device="cpu")


def test_encode_shape_and_rate(codec):
    wav = torch.sin(2 * torch.pi * 220 * torch.arange(24_000) / 24_000)  # 1 s tone
    codes = codec.encode(wav)
    assert codes.shape[0] == 32                  # 32 codebooks
    assert 11 <= codes.shape[1] <= 14            # ~12.5 frames per second
    assert codes.dtype == torch.long
    assert codes.max() < 2048 and codes.min() >= 0


def test_roundtrip_produces_audio(codec):
    wav = torch.sin(2 * torch.pi * 220 * torch.arange(24_000) / 24_000)
    out = codec.decode(codec.encode(wav))
    assert out.ndim == 1
    assert abs(out.shape[0] - 24_000) < 4_000    # roughly 1 s back
    assert out.abs().max() > 0.01                # not silence
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pip install -r requirements.txt && .venv/bin/pytest tests/test_codec.py -v`
Expected: FAIL — `ModuleNotFoundError: gemma_csm.codec`.

**Step 3: Implement**

`gemma_csm/codec.py`:
```python
import torch
from transformers import MimiModel


class MimiCodec:
    """Frozen Mimi: waveform <-> (32, T) integer codes. 24 kHz mono in/out."""

    def __init__(self, model_id: str = "kyutai/mimi", device: str = "cpu"):
        self.device = device
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (samples,) float in [-1, 1] at 24 kHz -> (32, frames) long."""
        x = wav.to(self.device).reshape(1, 1, -1)
        codes = self.model.encode(x).audio_codes  # (1, 32, T)
        return codes[0].cpu().long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (32, frames) long -> (samples,) float at 24 kHz."""
        x = codes.to(self.device).unsqueeze(0)    # (1, 32, T)
        wav = self.model.decode(x).audio_values   # (1, 1, samples)
        return wav[0, 0].cpu()
```

**Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_codec.py -v`
Expected: 2 PASS (first run downloads ~400 MB Mimi weights).

**Step 5: Human listen check (real speech, not a tone)**

Run:
```bash
.venv/bin/python - <<'EOF'
import torch, torchaudio, soundfile as sf
from gemma_csm.codec import MimiCodec
url = "https://upload.wikimedia.org/wikipedia/commons/transcoded/0/03/LL-Q1860_%28eng%29-Vealhurl-test.wav/LL-Q1860_%28eng%29-Vealhurl-test.wav.mp3"
# any short local speech file also fine; else record one
wav, sr = torchaudio.load(torchaudio.utils.download_asset(url) if url.startswith("http") else url)
wav = torchaudio.functional.resample(wav.mean(0), sr, 24_000)
c = MimiCodec()
sf.write("samples/roundtrip.wav", c.decode(c.encode(wav)).numpy(), 24_000)
print("wrote samples/roundtrip.wav")
EOF
open samples/roundtrip.wav
```
Expected: clearly intelligible speech. If the download asset is awkward, substitute any local speech file — the point is a human hears intelligible Mimi-roundtripped speech before we build on it.

**Step 6: Commit**

```bash
git add gemma_csm/codec.py tests/test_codec.py && git commit -m "feat: Mimi codec wrapper with verified roundtrip"
```

---

## Task 4: LJSpeech data prep

**Files:**
- Create: `gemma_csm/data_prep.py`, `tests/test_data_prep.py`, `gemma_csm/dataset.py`, `tests/test_dataset.py`

**Step 1: Write the failing test (prep on fake clips)**

`tests/test_data_prep.py`:
```python
import csv
import torch
import torchaudio

from gemma_csm.data_prep import prepare_shard


def test_prepare_shard(tmp_path):
    wav_dir = tmp_path / "wavs"; wav_dir.mkdir()
    for name in ("LJ001-0001", "LJ001-0002"):
        torchaudio.save(str(wav_dir / f"{name}.wav"),
                        torch.zeros(1, 22050), 22050)  # 1 s silence @ LJ rate
    meta = tmp_path / "metadata.csv"
    with open(meta, "w") as f:
        f.write("LJ001-0001|raw|Printing, then.\nLJ001-0002|raw|A second clip.\n")

    out = prepare_shard(meta, wav_dir, tmp_path / "out", limit=None)
    items = torch.load(out, weights_only=True)
    assert len(items) == 2
    assert items[0]["codes"].shape[0] == 32
    assert isinstance(items[0]["text"], str) and items[0]["text"] == "Printing, then."
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_data_prep.py -v`
Expected: FAIL — module not found.

**Step 3: Implement**

`gemma_csm/data_prep.py`:
```python
"""Encode LJSpeech to Mimi codes once; training never touches raw audio.

Usage (substrate):
  python -m gemma_csm.data_prep --root data/LJSpeech-1.1 --out data/shards [--limit 50]
"""
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from gemma_csm.codec import MimiCodec

LJ_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def prepare_shard(metadata_csv: Path, wav_dir: Path, out_dir: Path,
                  limit: int | None = None, device: str = "cpu") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device=device)
    items = []
    lines = Path(metadata_csv).read_text().strip().split("\n")
    if limit:
        lines = lines[:limit]
    for line in tqdm(lines, desc="encoding"):
        clip_id, _raw, normalized = line.split("|", 2)
        wav, sr = torchaudio.load(str(Path(wav_dir) / f"{clip_id}.wav"))
        wav = torchaudio.functional.resample(wav.mean(0), sr, 24_000)
        items.append({"id": clip_id, "text": normalized.strip(),
                      "codes": codec.encode(wav)})
    out = out_dir / (f"ljspeech_{len(items)}.pt")
    torch.save(items, out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    prepare_shard(a.root / "metadata.csv", a.root / "wavs", a.out,
                  limit=a.limit, device=a.device)
```

**Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_data_prep.py -v`
Expected: PASS.

**Step 5: Dataset class — failing test**

`tests/test_dataset.py`:
```python
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
```

**Step 6: Run to verify it fails, then implement**

`gemma_csm/dataset.py`:
```python
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
        ids = self.tok(it["text"], add_special_tokens=True)["input_ids"]
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
```

Run: `.venv/bin/pytest tests/test_dataset.py -v` → PASS.

**Step 7: Commit, then run real prep on substrate**

```bash
git add gemma_csm/data_prep.py gemma_csm/dataset.py tests/ && git commit -m "feat: data prep and dataset/collator"
scripts/sync-to-substrate.sh
```

On substrate (download ~2.6 GB, encode ~1–2 h on GPU; run in tmux or background):
```bash
ssh gianni@substrate
cd ~/gemma-csm && mkdir -p data
curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar xj -C data
.venv/bin/python -m gemma_csm.data_prep --root data/LJSpeech-1.1 --out data/shards --limit 50   # M0a shard, quick
.venv/bin/python -m gemma_csm.data_prep --root data/LJSpeech-1.1 --out data/shards              # full, slow
```
Expected: `data/shards/ljspeech_50.pt` then `ljspeech_13100.pt`. NOTE: GPU is mostly idle for Ollama at encode time but Mimi is small — safe to run alongside services.

---

## Task 5: Model — depth decoder (CPU-testable)

**Files:**
- Create: `gemma_csm/model.py`, `tests/test_depth_decoder.py`

**Step 1: Failing test**

`tests/test_depth_decoder.py`:
```python
import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import DepthDecoder


def test_depth_decoder_shapes_and_loss():
    cfg = GemmaCSMConfig()
    dd = DepthDecoder(cfg, backbone_dim=512)
    n_frames = 6
    h = torch.randn(n_frames, 512)                       # backbone hidden per frame
    codes = torch.randint(0, 2048, (n_frames, 32))       # full frame codes
    loss = dd.loss(h, codes)
    assert loss.ndim == 0 and loss.item() > 0


def test_depth_decoder_generate():
    cfg = GemmaCSMConfig()
    dd = DepthDecoder(cfg, backbone_dim=512)
    h = torch.randn(1, 512)
    c0 = torch.randint(0, 2048, (1,))
    frame = dd.generate(h, c0)                           # (1, 32)
    assert frame.shape == (1, 32)
    assert (frame[:, 0] == c0).all()
```

**Step 2: Run to verify failure, then implement**

`gemma_csm/model.py` (part 1 — DepthDecoder; GemmaCSM comes in Task 6):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma_csm.config import GemmaCSMConfig


def _causal_block(cfg: GemmaCSMConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.depth_d_model, nhead=cfg.depth_heads,
        dim_feedforward=cfg.depth_d_model * 4, batch_first=True,
        norm_first=True, dropout=0.0,
    )
    return nn.TransformerEncoder(layer, num_layers=cfg.depth_layers)


class DepthDecoder(nn.Module):
    """Per-frame: from backbone hidden h_t and c0, autoregressively predict c1..c31.

    Sequence per frame: [proj(h_t), emb_0(c0), emb_1(c1), ..., emb_30(c30)]
    position k predicts codebook k+1 (k = 0..30 predicts c1..c31... position 0
    is proj(h_t) predicting c0? No: c0 comes from the backbone head. Position 0
    (proj(h_t)+emb(c0)) predicts c1, position k predicts c_{k+1}.)
    """

    def __init__(self, cfg: GemmaCSMConfig, backbone_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.depth_d_model
        self.h_proj = nn.Linear(backbone_dim, d)
        # one embedding table per codebook level 0..30 (inputs)
        self.embs = nn.ModuleList(
            [nn.Embedding(cfg.codebook_size, d) for _ in range(cfg.n_codebooks - 1)]
        )
        self.pos = nn.Parameter(torch.zeros(cfg.n_codebooks - 1, d))
        self.tf = _causal_block(cfg)
        # one output head per predicted level 1..31
        self.heads = nn.ModuleList(
            [nn.Linear(d, cfg.codebook_size) for _ in range(cfg.n_codebooks - 1)]
        )

    def _inputs(self, h, codes_in):
        # h: (N, backbone_dim); codes_in: (N, K) levels 0..K-1 -> (N, K, d)
        toks = torch.stack(
            [self.embs[k](codes_in[:, k]) for k in range(codes_in.shape[1])], dim=1
        )
        return self.h_proj(h).unsqueeze(1) + toks + self.pos[: codes_in.shape[1]]

    def loss(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """h: (N, backbone_dim), codes: (N, 32). Teacher-forced CE over c1..c31."""
        K = self.cfg.n_codebooks - 1
        x = self._inputs(h, codes[:, :K])                      # predicts levels 1..K
        mask = nn.Transformer.generate_square_subsequent_mask(K, device=x.device)
        y = self.tf(x, mask=mask, is_causal=True)              # (N, K, d)
        losses = [
            F.cross_entropy(self.heads[k](y[:, k]), codes[:, k + 1])
            for k in range(K)
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate(self, h: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        """h: (N, backbone_dim), c0: (N,) -> full frames (N, 32)."""
        K = self.cfg.n_codebooks - 1
        out = [c0]
        for k in range(K):
            x = self._inputs(h, torch.stack(out, dim=1))
            mask = nn.Transformer.generate_square_subsequent_mask(k + 1, device=x.device)
            y = self.tf(x, mask=mask, is_causal=True)
            out.append(self.heads[k](y[:, -1]).argmax(-1))
        return torch.stack(out, dim=1)
```

**Step 3: Run tests** → `.venv/bin/pytest tests/test_depth_decoder.py -v` → 2 PASS.

**Step 4: Commit** — `git add -A && git commit -m "feat: depth decoder"`

---

## Task 6: Model — GemmaCSM wrapper (backbone + audio embeds + c0 head)

**Files:**
- Modify: `gemma_csm/model.py` (append)
- Create: `tests/test_model_cpu.py`

The CPU test uses a TINY randomly-initialized Gemma-architecture backbone built from a config (no download, no quantization) — it tests OUR wiring, not Google's weights. The real e2b loader is exercised on substrate in Task 7.

**Step 1: Failing test**

`tests/test_model_cpu.py`:
```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def tiny_backbone():
    # Llama stands in for Gemma4 text stack on CPU: same causal-LM interface
    # (inputs_embeds, output_hidden_states). Our wrapper only uses that interface.
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=512)
    return LlamaForCausalLM(cfg)


def make_model():
    cfg = GemmaCSMConfig(depth_d_model=64, depth_layers=1, depth_heads=2)
    return GemmaCSM(cfg, backbone=tiny_backbone()), cfg


def test_forward_loss():
    model, cfg = make_model()
    batch = {
        "text_ids": torch.randint(0, 500, (2, 7)),
        "text_mask": torch.ones(2, 7, dtype=torch.bool),
        "codes": torch.randint(0, 2048, (2, 32, 10)),
        "frame_mask": torch.ones(2, 10, dtype=torch.bool),
    }
    out = model(batch)
    assert out["loss_c0"].item() > 0 and out["loss_depth"].item() > 0


def test_generate_smoke():
    model, cfg = make_model()
    codes = model.generate_codes(torch.randint(0, 500, (1, 5)), max_frames=4)
    assert codes.shape[0] == 32 and 1 <= codes.shape[1] <= 4
```

**Step 2: Run to verify failure, then implement (append to `gemma_csm/model.py`)**

```python
class GemmaCSM(nn.Module):
    """Backbone (any HF causal LM) + audio frame embeddings + c0 head + DepthDecoder.

    Sequence: [text tokens][audio frames...]. Audio frame t input embedding =
    sum over 32 codebook embeddings. Backbone hidden at the LAST TEXT position
    predicts frame 0's c0; hidden at frame t predicts frame t+1's c0 (shifted
    next-frame prediction, standard causal LM but in frame space). EOS class
    (id 2048) on the c0 head terminates generation.
    """

    def __init__(self, cfg: GemmaCSMConfig, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        d_model = backbone.config.hidden_size
        self.audio_embs = nn.ModuleList(
            [nn.Embedding(cfg.codebook_size, d_model) for _ in range(cfg.n_codebooks)]
        )
        self.c0_head = nn.Linear(d_model, cfg.codebook_size + 1)  # +1 = EOS
        self.depth = DepthDecoder(cfg, backbone_dim=d_model)

    # -- embedding assembly ---------------------------------------------
    def _embed_frames(self, codes):
        # codes: (B, 32, T) -> (B, T, d)
        return torch.stack(
            [self.audio_embs[k](codes[:, k]) for k in range(self.cfg.n_codebooks)]
        ).sum(0)

    def _embed_text(self, text_ids):
        return self.backbone.get_input_embeddings()(text_ids)

    # -- training forward -------------------------------------------------
    def forward(self, batch):
        text_e = self._embed_text(batch["text_ids"])             # (B, Tt, d)
        audio_e = self._embed_frames(batch["codes"])             # (B, Tf, d)
        x = torch.cat([text_e, audio_e], dim=1)
        attn = torch.cat([batch["text_mask"], batch["frame_mask"]], dim=1)
        h = self.backbone(
            inputs_embeds=x, attention_mask=attn, output_hidden_states=True
        ).hidden_states[-1]

        B, Tt = batch["text_ids"].shape
        Tf = batch["codes"].shape[2]
        # positions predicting frame t's c0: last-text (t=0) .. frame Tf-1 shifted
        pred_h = h[:, Tt - 1 : Tt + Tf - 1]                      # (B, Tf, d)

        # c0 targets: real c0 where frame has content beyond EOS slot, EOS at slot
        c0_tgt = batch["codes"][:, 0].clone()                    # (B, Tf)
        eos_pos = batch["frame_mask"].sum(1) - 1                 # index of EOS slot
        c0_tgt[torch.arange(B), eos_pos] = self.cfg.c0_eos_id
        logits = self.c0_head(pred_h)
        loss_c0 = F.cross_entropy(
            logits[batch["frame_mask"]], c0_tgt[batch["frame_mask"]]
        )

        # depth loss on a random fraction of REAL frames (not the EOS slot)
        real = batch["frame_mask"].clone()
        real[torch.arange(B), eos_pos] = False
        idx_b, idx_t = real.nonzero(as_tuple=True)
        n_keep = max(1, int(len(idx_b) * self.cfg.depth_loss_frame_frac))
        sel = torch.randperm(len(idx_b))[:n_keep]
        ib, it = idx_b[sel], idx_t[sel]
        frame_h = h[ib, Tt + it]                                  # hidden AT frame t
        frame_codes = batch["codes"][ib, :, it]                   # (n, 32)
        loss_depth = self.depth.loss(frame_h, frame_codes)

        return {"loss_c0": loss_c0, "loss_depth": loss_depth,
                "loss": loss_c0 + loss_depth}

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def generate_codes(self, text_ids, max_frames: int = 1000, temperature: float = 0.9):
        x = self._embed_text(text_ids)                            # (1, Tt, d)
        frames = []
        for _ in range(max_frames):
            h = self.backbone(
                inputs_embeds=x, output_hidden_states=True
            ).hidden_states[-1][:, -1]                            # (1, d)
            logits = self.c0_head(h) / temperature
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)                    # (1, 32)
            frames.append(frame[0])
            x = torch.cat([x, self._embed_frames(frame.unsqueeze(-1))], dim=1)
        if not frames:
            return torch.zeros(32, 1, dtype=torch.long)
        return torch.stack(frames, dim=1)                         # (32, T)
```

NOTE for implementer: `generate_codes` recomputes the full prefix each frame (no KV cache). Fine for M0 eval (125 frames ≈ seconds of compute); do NOT optimize yet — YAGNI.

**Step 3: Run all CPU tests** → `.venv/bin/pytest -v` → all PASS.

**Step 4: Commit** — `git add -A && git commit -m "feat: GemmaCSM wrapper with dual loss and frame generation"`

---

## Task 7: Real backbone loader (4-bit + LoRA, substrate)

**Files:**
- Create: `gemma_csm/loader.py`, `tests/test_loader_gpu.py`

**Step 1: Implement loader**

`gemma_csm/loader.py`:
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def load_gemma_csm(cfg: GemmaCSMConfig, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    # text-only path: if the checkpoint is the multimodal wrapper, unwrap it
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    backbone = get_peft_model(backbone, lora)
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
    model = GemmaCSM(cfg, backbone=backbone)
    # new modules in bf16 on GPU; quantized backbone manages its own placement
    model.audio_embs.to(device, torch.bfloat16)
    model.c0_head.to(device, torch.bfloat16)
    model.depth.to(device, torch.bfloat16)
    return model, tok
```

`tests/test_loader_gpu.py`:
```python
import pytest
import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm


@pytest.mark.gpu
def test_load_and_forward_fits_16gb():
    model, tok = load_gemma_csm(GemmaCSMConfig())
    ids = torch.tensor([tok("Hello world.")["input_ids"]], device="cuda")
    batch = {
        "text_ids": ids,
        "text_mask": torch.ones_like(ids, dtype=torch.bool),
        "codes": torch.randint(0, 2048, (1, 32, 25), device="cuda"),
        "frame_mask": torch.ones(1, 25, dtype=torch.bool, device="cuda"),
    }
    out = model(batch)
    out["loss"].backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {peak_gb:.1f} GB")
    assert out["loss"].item() > 0
    assert peak_gb < 14.0
```

**Step 2: Run on substrate (GPU must be free — stop services first, see Task 8 scripts; or check `nvidia-smi` headroom)**

Run:
```bash
scripts/sync-to-substrate.sh
ssh gianni@substrate 'cd ~/gemma-csm && .venv/bin/pytest tests/test_loader_gpu.py -v -m gpu -s'
```
Expected: PASS, printed peak under ~12 GB. First run downloads ~6 GB of e2b weights.
LIKELY FAILURE POINTS (debug here, this is the riskiest task):
- Gemma 4 checkpoint class/unwrap shape — inspect `type(backbone)`, adjust the `language_model` unwrap.
- dtype mismatches between bf16 new modules and 4-bit backbone output — cast `h` in `GemmaCSM.forward` if needed.

**Step 3: Commit** — `git add -A && git commit -m "feat: 4-bit QLoRA e2b loader, verified on 4060 Ti"`

---

## Task 8: Service stop/start + training script

**Files:**
- Create: `scripts/gpu-claim.sh`, `scripts/gpu-release.sh`, `gemma_csm/train.py`

**Step 1: Discover Hermes unit name on substrate**

Run: `ssh gianni@substrate 'systemctl list-units --type=service | grep -iE "hermes|ollama" ; pgrep -af hermes | head -3'`
Write the discovered names into the scripts below (placeholder `HERMES_UNIT`). If Hermes is not a systemd unit (e.g. tmux/pm2), adapt: the scripts must stop/start it however it actually runs — verify with the user if ambiguous. DO NOT touch other services.

**Step 2: Write scripts**

`scripts/gpu-claim.sh` (on substrate, run before training):
```bash
#!/usr/bin/env bash
set -euo pipefail
sudo systemctl stop ollama
# HERMES: replace with discovered mechanism, e.g.: sudo systemctl stop hermes
nvidia-smi --query-gpu=memory.used --format=csv
```

`scripts/gpu-release.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
sudo systemctl start ollama
# HERMES: restart equivalent
nvidia-smi --query-gpu=memory.used --format=csv
```

**Step 3: Training script**

`gemma_csm/train.py`:
```python
"""Train Gemma-CSM. M0a: --shard data/shards/ljspeech_50.pt --overfit

Run on substrate inside tmux:
  scripts/gpu-claim.sh
  .venv/bin/python -m gemma_csm.train --shard data/shards/ljspeech_50.pt \
      --steps 2000 --out checkpoints/m0a
  scripts/gpu-release.sh
"""
import argparse, time
from pathlib import Path

import torch
import soundfile as sf
from torch.utils.data import DataLoader

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset import CodesDataset, collate
from gemma_csm.loader import load_gemma_csm

EVAL_SENTENCES = [
    "Printing, in the only sense with which we are at present concerned.",
    "The quick brown fox jumps over the lazy dog.",
    "This voice was generated by a Gemma backbone.",
]


def save_samples(model, tok, codec, out_dir: Path, step: int):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(EVAL_SENTENCES):
        ids = torch.tensor([tok(s)["input_ids"]], device="cuda")
        codes = model.generate_codes(ids, max_frames=200)
        wav = codec.decode(codes)
        sf.write(out_dir / f"step{step:06d}_s{i}.wav", wav.float().numpy(), 24_000)
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("checkpoints/run"))
    p.add_argument("--eval-every", type=int, default=500)
    a = p.parse_args()

    cfg = GemmaCSMConfig()
    model, tok = load_gemma_csm(cfg)
    codec = MimiCodec(device="cpu")  # decode for eval on CPU, keep VRAM clean
    ds = CodesDataset([a.shard], tok, max_frames=cfg.max_frames)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=cfg.lr)

    step, t0 = 0, time.time()
    while step < a.steps:
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(batch)
            (out["loss"] / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                opt.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"step {step}  c0 {out['loss_c0']:.3f}  "
                      f"depth {out['loss_depth']:.3f}  "
                      f"{(time.time()-t0)/max(step,1):.1f}s/step", flush=True)
            if step and step % a.eval_every == 0:
                save_samples(model, tok, codec, a.out / "samples", step)
                torch.save({k: v for k, v in model.state_dict().items()
                            if "backbone.base_model" not in k or "lora" in k},
                           a.out / f"ckpt_{step}.pt")
            step += 1
            if step >= a.steps:
                break
    save_samples(model, tok, codec, a.out / "samples", step)


if __name__ == "__main__":
    main()
```

**Step 4: Smoke the loop (200 steps, GPU)**

Run on substrate (services stopped): `.venv/bin/python -m gemma_csm.train --shard data/shards/ljspeech_50.pt --steps 200 --eval-every 200 --out checkpoints/smoke`
Expected: loss prints decreasing-ish c0 from ~7.6 (ln 2049) downward; sample WAVs appear (will sound like noise at step 200 — fine). Note s/step for run planning.

**Step 5: Commit** — `git add -A && git commit -m "feat: training loop with periodic sample generation"`

---

## Task 9: M0a — the overfit gate

**Step 1: Overnight run on substrate (tmux)**

```bash
ssh gianni@substrate
cd ~/gemma-csm && scripts/gpu-claim.sh
tmux new -s m0a
.venv/bin/python -m gemma_csm.train --shard data/shards/ljspeech_50.pt \
    --steps 6000 --eval-every 1000 --out checkpoints/m0a 2>&1 | tee m0a.log
# detach; next morning:
scripts/gpu-release.sh
```

**Step 2: Listen**

Run (Mac): `rsync -av gianni@substrate:~/gemma-csm/checkpoints/m0a/samples/ ~/workspace/gemma-csm/samples/m0a/ && open ~/workspace/gemma-csm/samples/m0a/`
Note: EVAL_SENTENCES[0] is IN the 50-clip training set (memorization test); [1] and [2] are not (they will likely be garbled at M0a — that's expected and fine).

**Step 3: GATE decision**
- c0 loss well under ~1.0 on the overfit set AND sentence 0 clearly intelligible → **M0a PASSED**, commit log + a sample, proceed to M0b (full shard, same script, `--steps 40000`, nightly runs).
- Loss plateaus high or audio is structured noise → STOP. Debug wiring (masking off-by-one at the text/audio boundary is suspect #1, dtype casts #2). Do NOT start full-data runs until 50-clip memorization works.

**Step 4: Record result**

Append outcome (loss curve summary, s/step, verdict) to `docs/plans/2026-06-10-gemma-csm-design.md` under a new `## M0a result` heading. Commit.

---

## Task 10: M0b kickoff (only after M0a gate passes)

Same trainer, full shard: `--shard data/shards/ljspeech_13100.pt --steps 40000 --eval-every 2000 --out checkpoints/m0b`, run nightly between `gpu-claim`/`gpu-release`, listen to samples each morning. Add a val split + held-out loss before long runs (10% of shard; small change to `CodesDataset` — write the test first). Success criterion per design doc: held-out sentences intelligible to Gianni.
