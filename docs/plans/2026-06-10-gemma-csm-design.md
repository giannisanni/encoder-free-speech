# Gemma-CSM: CSM-style speech generation on a Gemma 4 backbone

**Date:** 2026-06-10
**Status:** Design approved, pre-implementation

## Thesis

Build a Sesame-CSM-style conversational speech model, but with Google's Gemma 4
(encoder-free, natively multimodal) as the backbone instead of a text-only Llama.

We are NOT trying to beat CSM/MisoTTS at reading text aloud — they trained on
~1M hours, we train on 24. The bet is that a Gemma backbone unlocks what their
architecture cannot: speech generation conditioned on *understood* audio/video
input, stronger language reasoning, and a latent multilingual path. Milestone-0
is a cheap feasibility probe of the one unproven assumption: that Gemma's
pretrained representations transfer to *generating* Mimi audio codes via QLoRA.

## Architecture (hierarchical, CSM-faithful)

```
text (+ later: audio/video context)
        |
        v
+----------------------+
| Gemma 4 e2b backbone |  4-bit NF4 + LoRA. 1 step per audio frame (12.5 Hz)
| (5.1B total / 2.3B   |--> new linear head -> codebook-0 token (coarse)
|  effective, PLE)     |
+----------------------+
     hidden state h_t
        |
        v
+----------------------+
| depth decoder ~150M  |  bf16, trained from scratch (CSM scaffold)
| autoregressive in    |--> codebooks 1..31 for frame t
| codebook dimension   |
+----------------------+
        |
        v
  Mimi.decode (frozen) -> 24 kHz WAV
```

- **Audio tokenizer:** kyutai Mimi, frozen. 12.5 Hz frames, 32 RVQ codebooks,
  2048-entry vocab each. 10 s of speech = 125 backbone steps.
- **Frame input embedding:** sum of 32 codebook embeddings from a NEW audio
  embedding table (32 x 2048 x d_model). Text uses Gemma's native embeddings.
- **Sequence format (M0):** `[BOS] <text> [SPEAKER] <audio frames...> [AUDIO-EOS]`
- **Inference:** autoregress frames; backbone -> c0, depth decoder -> c1..c31,
  re-embed full frame, feed back. Stop at AUDIO-EOS, Mimi-decode.

## Training strategy

- **Trainables:** LoRA adapters on backbone, audio embedding table, c0 head,
  depth decoder. Backbone base weights frozen in 4-bit. Mimi frozen.
- **Losses:** CE on codebook-0 (backbone head) + CE on codebooks 1-31 (depth
  decoder, computed on a random 1/16 frame subset per batch — CSM's memory trick).
- **Stack:** plain `transformers` + `peft` + `bitsandbytes`. Deliberately NOT
  Unsloth: its optimizations assume standard architectures and will fight our
  custom forward. Debuggability > 30% speed on a research project.
- **Memory budget (4060 Ti 16 GB):** backbone 4-bit ~2.6 GB, LoRA+optimizer
  ~0.3 GB, audio embeds+head ~1-2 GB, depth decoder ~2 GB, activations (batch
  2-4, grad checkpointing) ~3-5 GB. Total ~9-12 GB. Fits.
- **Config flag `n_codebooks`:** default 32 (CSM-faithful); 8 available for
  ~4x faster depth-decoder iteration if M0b converges slowly. All 32 codebooks
  are precomputed regardless, so flipping the flag never re-runs data prep.

## Data

- **Dataset (M0):** LJSpeech — 24 h, 13k clips, single female English speaker,
  public domain. Chosen to make architecture the only variable.
- **Pipeline (one-time ~1-2 h):** download -> resample 24 kHz mono ->
  Mimi.encode all clips -> save codes + Gemma-tokenized text as .pt shards.
  Training never touches raw audio.

## Compute & operations

- **Hardware:** substrate (RTX 4060 Ti 16 GB, CUDA 12, Python 3.12).
- **GPU windows:** Ollama + Hermes are stopped during training runs (overnight,
  ~8 h), restarted after. Hermes survives on its Ollama Cloud primary.
- **Estimate (e2b):** ~1-2 h/epoch on full LJSpeech; intelligibility expected
  somewhere in 30-100 epochs => ~50-150 GPU-hours => 1-2 weeks of nights.
  The epoch count is the genuinely uncertain number; the first run's loss curve
  recalibrates it.
- **12B stage:** out of scope until e2b proves the concept; likely needs a
  rented 24-48 GB GPU.

## Milestone ladder

| # | Goal | Gate |
|---|------|------|
| M0a | Overfit 50 clips, parrot them intelligibly | Hard gate: if it can't memorize 50 clips in one night, debug wiring HERE (shape/masking bugs die in minutes-scale iteration, not on full runs) |
| M0b | Full LJSpeech, intelligible novel sentences, one voice | Human (Gianni) understands held-out sentences; val loss + periodic sample WAVs |
| M1 | Voice cloning from reference audio context | CSM-style context conditioning |
| M2 | Audio/video-conditioned speech — the thesis cash-out | Things CSM/Miso architecturally cannot do |

- **Eval during training:** every N steps generate 3 fixed test sentences,
  copy WAVs to the Mac for listening.

## Code reuse

| Source | License | What we take |
|--------|---------|--------------|
| SesameAILabs/csm | Apache-2.0 | Model scaffold: frame embedding, depth decoder, per-frame sampling loop (swap Llama -> Gemma) |
| HF transformers | Apache-2.0 | CSM class as reference, Gemma 4 loading, Mimi codec |
| MisoTTS | Modified MIT (attribution above 50M MAU / $10M-mo — n/a) | Second reference implementation of the same architecture |
| We write | — | Gemma<->audio-head glue, dataset/collator, training loop, eval scripts |

## Key risks

1. **Transfer risk (the bet itself):** Gemma representations may transfer
   poorly to audio generation under QLoRA. M0a/M0b exist to learn this cheaply.
2. **Silent wiring bugs:** most likely failure mode; mitigated by the strict
   M0a overfit gate.
3. **Epoch-count uncertainty:** 3 nights to 3 weeks; first loss curve decides.
4. **GPU contention:** training competes with Hermes/Ollama; mitigated by
   overnight windows and service stop/start scripts.
