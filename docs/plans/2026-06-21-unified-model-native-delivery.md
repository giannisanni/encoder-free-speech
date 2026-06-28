# Unified Model via Native Audio Delivery — Implementation Plan

**Goal:** Collapse the working two-pass cascade into one end-to-end network by
delivering the query audio through Gemma-4's *native* multimodal forward and
conditioning the CSM speech head on the resulting hidden states — not on a
hand-built feature-splice and not on re-encoded reply text.

**Approach:** Qwen2.5-Omni "Thinker–Talker" style (pragmatic, recommended). The
frozen backbone autoregressively produces the reply *and* its hidden states from
the native audio (Thinker); the CSM head speaks from the reply-position hidden
states (Talker). Backbone stays frozen + LoRA; only the head + LoRA train. Text is
still emitted internally (fine — Qwen does the same); the win is removing the
text re-encode bottleneck so input prosody can flow into the reply.

**Why the current unified model fails (root cause):**
1. **Delivery (Bug A):** `generate_unified`/`forward_unified` splice projected
   audio embeds into the *extracted* `language_model` (`self.backbone(inputs_embeds=…)`),
   bypassing the full model's `masked_scatter` + per-layer-embedding (PLE)
   handling for `<audio>` positions. That path carries prosody, not lexical
   content. Proven by zeroed-audio ablation: greedy output identical with real vs
   zeroed audio.
2. **Objective (Bug B):** the joint layout teacher-forced reply text with no
   dropout, so the loss rewarded *vocalizing a supplied answer*, not *deciding*
   one. Native delivery (Bug A fix) is what makes the text objective meaningful —
   the reply loss can then only be satisfied by attending to the audio.
3. **Structural:** a frozen trunk + external head reading a content-less splice has
   no gradient path to put content into the hidden states. Native delivery puts it
   there; then frozen+LoRA may suffice.

**Key fact established:** `input_features` in the shards are produced by the real
Gemma `AutoProcessor` (`data_prep_opens2s.py:175`). So the *only* difference
between content-carrying and content-losing is the forward route. Same features,
two routes — that is what the Phase 0 probe isolates.

---

## Phase 0 — Delivery-fix probe (GATING; no code changes to model.py)

**File:** `scripts/exp_native_delivery.py` (written).

**What it does:** loads the full `Gemma4UnifiedForConditionalGeneration` in the
hybrid quant, takes a held-out OpenS2S item, builds the native input
(`_UNI_PREFIX + <audio>*N + _UNI_SUFFIX`, N = `get_audio_features` token count),
and routes through `full.generate(input_ids=…, input_features=…)`. Then:
- **(A)** real vs zeroed audio → does the reply text change and is the real reply
  on-topic? (splice did *not* change.)
- **(B)** teacher-forces the reply, reads `output_hidden_states[-1]` at the reply
  positions, compares norm real vs zeroed → do the head-input states depend on the
  audio?
- Introspects `full.forward` signature → the exact audio kwarg names the
  `forward_unified` rewrite must use.

**Run:**
```
GEMMA_BACKBONE=google/gemma-4-12B-it \
.venv/bin/python -m scripts.exp_native_delivery --shard <heldout>.pt --index 0
```

**Gate:**
- BOTH (A) and (B) pass → delivery was the bug; proceed to Phase 1.
- Either fails → stop. Investigate placeholder/framing/PLE alignment first
  (the probe's introspection output tells us where). Do **not** build the
  training route on an ungrounded forward.

**Cost:** ran free on substrate (RTX 4060 Ti, model cached) — not a pod.

### Phase 0 RESULT (2026-06-21): PASS — delivery was the bug.
Ran `opens2s_en_000.pt` item 0 (query: *"Why does it feel like no one understands
me?"*) through the full native forward on substrate.
- **(A) PASS:** real-audio reply is on-topic/empathetic ("…it's often not a
  reflection of who you are… communication is a two-way street…"); zeroed-audio
  reply is generic ("the is a definite article…"). The splice gave *identical*
  output regardless of audio; native delivery does not.
- **(B) PASS** (after fixing the metric — norm was wrong): reply-position hidden
  states differ real-vs-zeroed by **cosine-dist 0.041, rel-L2 24.5%, next-token
  disagreement 18.5%**. (The original norm-delta read only 1% — magnitudes are
  similar, directions differ a lot; norm can't see it.)
- **Peak VRAM 8.16 GB** on the 16 GB target card — ample headroom for head + LoRA.
- **Mechanism confirmed:** the splice fed the *extracted* LM pre-projected audio
  embeds with NO per-layer embeddings (PLE) at the audio positions. The full model
  computes PLE from the `<audio>` placeholder ids + runs `masked_scatter`; that is
  what carries lexical content. `forward_unified` must let the full model own this.

### Refinement for Phase 1 (learned here)
The forward can't be one clean `full(input_ids=…)` call: the sequence mixes text
(framing + reply), **native audio-IN** (`input_features` at `<audio>` placeholders),
and **audio-OUT frames** (custom Mimi-code embeds, not in the text vocab). Split it:
1. Run the **full** model over `[framing | <audio>*N | suffix | reply_text | <eos>]`
   with `input_features` (kwarg confirmed: `input_features` / `input_features_mask`),
   `output_hidden_states=True`. Text LM loss on reply positions; the `<eos>` hidden
   state seeds the head.
2. Continue the **frame** positions through `self.backbone` (== the full model's
   `language_model`, so KV-compatible) with the step-1 `past_key_values`; the head
   reads those hidden states for c0/depth loss. Mirrors how generation already works.

---

## Phase 1 — `forward_unified` rewrite: head on native hidden states

**Files:** `gemma_csm/model.py`, `gemma_csm/loader.py`, `gemma_csm/train_audio_in.py`.

**1.1 Loader exposes the full model.** `load_gemma_csm_audio_in` currently keeps
`mm` alive only inside the `audio_encode` closure. Return the full model (or a
forward closure that accepts `input_ids` + `input_features` + `output_hidden_states`)
so `GemmaCSM` can run the native forward, LoRA-adapted, with grad on the language
layers. Verify LoRA still targets the language decoder layers when called via the
full wrapper (PLE + adapters must stay on the accelerated path).

**1.2 New training forward.** Sequence:
`[ _UNI_PREFIX | <audio>*N | _UNI_SUFFIX | REPLY_TEXT | <eos> | REPLY_AUDIO ]`,
delivered to the **full** model with `input_features` (native masked_scatter),
`output_hidden_states=True`. Losses:
- **text LM** on `REPLY_TEXT` positions (teacher-forced target = the dataset reply)
  — this is the Thinker objective: learn to produce the reply *from the audio*.
- **c0 + depth** on `REPLY_AUDIO` positions, with the head conditioning on the
  hidden states of the `REPLY_TEXT` positions (+ the handoff/BOS state).
Keep the BOS-position calibration fix (attention-sink NaN) on the new geometry.

**1.3 Geometry unit test (CPU).** Extend `tests/test_forward_unified.py`: fake
full-model forward that accepts `input_features` + `<audio>` ids and returns
hidden states; assert losses + grads finite over the new geometry.

**Verify:** CPU test passes; a 1-step GPU forward/backward runs at < 16 GB peak
(the native forward + LoRA grad ≈ the 13.7 GB the splice forward already hit).

### Phase 1 STATUS (2026-06-21): forward + loader DONE and validated on substrate.
- `loader.load_gemma_csm_audio_in` now sets `model.full_model` + `model.audio_token_id`.
- `model.forward_unified` rewritten: **seg1** = full model over
  `[framing | <audio>*N | framing | reply_text | <eos>]` with `input_features`
  (native content delivery) + text LM loss; **seg2** = backbone over the reply
  frames using seg1's `past_key_values` → c0/depth (head conditions on hidden
  states from the native audio context). Loss geometry identical to the old splice.
- `scripts/val_unified_forward.py`: 1-step fwd/bwd on a real OpenS2S item →
  **PASS** (loss_text 2.35, c0 8.86, depth 8.69 finite; grads to 835/844 trainables;
  **13.36 GB** at n=1/maxframes60). n=2 OOMs → **batch 1 + grad-accum** is the rule.
- Constraint locked in: **gradient checkpointing OFF** (seg1 needs `use_cache`).

**Remaining for Phase 1:** (a) wire `train_audio_in.py --unified` to pass raw
`input_features` (skip the `audio_encode` pre-splice), force GC off, batch 1;
(b) rewrite `generate_unified` for native delivery (free-running decode + the
zeroed-audio ablation gate). CPU `test_forward_unified.py` can't cover native
delivery — the substrate 1-step check is the real test.

---

## Phase 2 — Train + free-running eval (gate on WER, never teacher-forced loss)

**2.1 Short overfit** on ~50 pairs: confirm the free-running decode (no reply text
supplied) produces an on-topic spoken reply, and re-run the zeroed-audio ablation
on the *trained* model — output must change with audio.

### Phase 2.1 RESULT (2026-06-21): grounding CONFIRMED; audio quality needs the full run.
Overfit 16 items, 800 steps, warm-start `tts_speak/ckpt_76000` (LOCAL — head_v2's
4.2 GB HF download filled substrate's disk to 268 MB; recovered). Teacher-forced
losses converged hard (text 10.5→0.000, c0→0.001). Free-running decode (the real
gate):
- **Grounding ✓:** reply TEXT tracks the audio and is item-specific (e.g. "why
  does no one understand me" → "It's okay to feel that way… I'm here to listen");
  the zeroed-audio ablation returns one canned generic line for every item → the
  model decides what to say *from the speech*, no bypass (3/4 clean; 1 degenerated
  into word-salad text from over-sampling a 16-item set).
- **Audio ✗ (expected):** whisper can't recover the text from the generated speech
  — c0≠intelligibility gap again (teacher-forced c0→0, free-running garbled).
  Causes, both addressed by the full run: only 800 steps + a *mismatched*
  single-voice TTS warm-start (not the hybrid-trained head_v2).
**Verdict:** the native-delivery unified ARCHITECTURE is validated end-to-end (it
learns + grounds). Intelligible audio needs Phase 2.2: head_v2 warm-start + the
full OpenS2S data + 8–16k steps. Substrate's chronic disk tightness (head_v2
download alone fills it) makes the A6000 the right venue for 2.2.

**2.2 Scale run** (8k–16k steps, A6000, warm-start the head from the cascade's
head_v2). Eval: 40 held-out OpenS2S queries → free-running decode → faster-whisper
→ WER + word-recall vs a correct reply, **and** the zeroed-audio ablation as a
standing regression gate.

**Success criterion:** grounded spoken replies (output depends on audio) at WER
comparable to the cascade (≈13%), in **one** forward pass. That is the unified
model.

---

## Phase 3 (optional, ambitious) — text-free (MOSS-Speech style)

Only if Phase 2 lands and we want *no* text anywhere: carve speech-prediction
layers into the trunk (MOSS splits the last 4 of 36 layers into parallel
text/speech stacks), train all four modality directions, and allow a **partial
unfreeze** (MOSS stage 2). Larger compute; separate plan.

---

## Risks / unknowns
- **PLE at audio positions:** Gemma-4 injects per-layer embeddings from `input_ids`.
  The native forward computes PLE for `<audio>` tokens; the splice did not. The
  probe confirms native delivery works end-to-end, but the `forward_unified`
  rewrite must let the full model own PLE (don't hand-build it).
- **Placeholder count alignment:** N must equal `get_audio_features` output length
  or `masked_scatter` errors. The probe derives N the same way.
- **Hidden-state distribution:** head_v2 was trained on extracted-LM hidden states;
  the full-model last hidden state may differ in scale. Warm-start then fine-tune;
  watch the first-step loss.
- **Memory:** native forward + LoRA grad must stay < 16 GB; if tight, the A6000
  (48 GB) is the training target anyway.
