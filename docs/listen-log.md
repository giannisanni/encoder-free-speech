# Listening log — perceptual evaluations by Gianni (the project's primary eval instrument)

Novel-sentence probe is s1: "The quick brown fox jumps over the lazy dog" (not in training data).

| Run / step | depth loss | Gianni's description | Interpretation |
|---|---|---|---|
| M0a v1, 1k | ~3.4 (leaky) | "scrambled, like a robot trying to produce speech", "speech played backwards on old tape recorders" | F1 leakage: semantic stream fine, acoustic garbage at inference |
| M0a v2, 1k | n/a | "a gasp", 0.2 s | F2 padding geometry: premature EOS |
| M0a v3, 4k | ~0.02 | full first clause "clear, sounded good", tail garbled ~0.1-0.2 s | memorization works; mild exposure-bias drift at tail |
| M0b, 8k | ~7.0 | "garbled mumbling, but I think I heard the last word fox if I listen closely" (position mismatch -> treated as pareidolia) | semantic stream generalizing, acoustic stream near-random |
| M0b, 22k | ~6.1 | "still mumbling but sounds better than the 8000 file; a few sounds that could potentially sound like the words if I listen closely" | perceptual improvement tracks depth-loss decline; proto-words emerging |
| M0b, 38k (~final) | ~6.5 |  "I kinda could make up 'the quick brown fox' but that is only because I know what it should say; still not understandable speech" (40k final: "verryyy slightly better") | PRIMED intelligibility reached on novel text; blind intelligibility not. Baseline ends acoustic-bottlenecked, as the flat depth loss predicted |
| M0b, 38k (detail) | ~6.5 | "the first part sounds better; I can barely make out: the quick brown f_s... ('the quick brown' is the most understandable part)" | Intra-utterance clarity gradient: (1) front-of-utterance best = exposure-bias drift; (2) clarity tracks word frequency in training data — "fox" occurs 0 times in LJSpeech and is the word that breaks, rendered as consonant skeleton (f_s) with mushy vowel; fricatives survive under-training better than vowel formants |

Pattern so far: c0 (semantic) loss governs speech-shaped rhythm and duration; depth (acoustic) loss governs word crispness. Perceptual reports have tracked depth loss monotonically.
| M0b-v2, 26k (66k cumulative) | ~6.45 (flat) | "slightly less mushy, I can ALMOST hear her say the full sentence; 'the quick brown' almost sound like real words; feels like it needs a bit more time" | A/B vs baseline-final: v2 audibly better DESPITE flat aggregate depth loss — perceptual gains likely concentrated in low (acoustically dominant) codebooks that the 31-codebook CE average hides. Third time perceptual eval out-resolved the loss curve |
| M0b-v2 FINAL, 40k global (80k cumulative) | ~6.3 | "the best one yet" | v2 ends as the project's perceptual peak: primed-intelligible novel speech, monotonically improved throughout. Handoff state for the multi-speaker run |
| m1_data (multi-speaker), 20k | ~6.75 | "step 20k is particularly good so far, the best out of the flipbook" | Within-run peak per listener at unremarkable depth CE — 4th observation of perception outrunning the depth metric; motivates the Whisper-WER harness |
| m1_data FINAL, 40k (cooled, temp 0.6/topk 50) vs v2 final | m1 depth ~6.7 | "I would honestly say the m0b_v2 sounds better" | KEY RESULT: multi-speaker training REGRESSED the focal LJ voice despite cooled sampling. Capacity dilution at fixed depth-transformer size: breadth bought with per-voice precision. Third experiment (more signal / more data / now capacity) — all point at depth-transformer capacity as binding constraint. v2 final = reigning single-voice champion |
| BLIND TEST #1 (v2 champion, cooled): "The farmer remembered a long letter after the storm." | depth ~6.3 | Heard: ".....after the....." (2/9 words, correct position) | First true blind intelligibility measurement: ~22%. M0 criterion NOT met. Quantifies the primed-listening illusion: primed reports were near-full-sentence; blind yields fragments. Protocol: sentence composed randomly on substrate, unseen by either party |
| BLIND TEST #2, double-blind A/B (2 novel sentences x both models, cooled, shuffled) | — | m1: "my father hates"≈farmer, "after this"≈after the (~14%); v2: "farmer" exact + one blank (~6%) | REVERSAL CONFIRMED: m1 beats v2 blind on novel text; v2's fox-probe win was memorized-text home advantage. m1's misses are phonetically adjacent (voice-pin stabilizes acoustics); v2 shows unconditioned voice-wobble ("echo/two voices" per listener). Both far below criterion; capacity run inherits m1's advantages. n=2 → WER harness required for statistical power |
| m1_cap 10k vs m1_data 10k (matched-step A/B) | both ~7.0 | "they sound equally bad" | Ear and CE agree for once (parity at matched steps); capacity verdict deferred to asymptote per design |
| m1_cap 12k vs m1_data 12k | cap ~7.0, m1 ~6.8 | "the m1 at 12000 sounds better than the m1cap at 12000" | Warm 150M still ahead of fresh 425M at matched steps; ear tracks CE within ~0.1 again (3rd consecutive agreement) |
| cap final fox (cooled, primed listen) | depth 6.76 | "actually pretty good for what it is, i can almost hear [the whole sentence] but with noise and distortion" | Words ~there, acoustics noisy — consistent with depth at codec noise floor; motivates codebook-truncation test (Whisper blind: 11% recall) |
| cap fox decoded at 32/16/8 codebooks (same tokens) | — | "all 3 sound the same" (caught "fox" in top16) | Deep codebooks add no audible noise at this quality level; truncation = compute lever only. Verdict: continue training (m2_sem) |
| m2_sem 2k+4k (reset-shock recovery, c0 5.0/4.4) | c0 well above m1 final 3.3 | "they genuenly sounf like shit lmao" | Expected: adapter-reset regression era; ear-CE agreement holds in the downward direction too |
| BLIND #4 (double-blind, m2 vs m1, 2 novel sentences) | m2 c0 ~3.1 vs m1 3.3 | A(m1):0/8, B(m2):"the...box larger" 2/8, C(m2):"winter" 1/8, D(m1):"warm" 0/8 | m2 ~19% vs m1 ~0-6%; both m2 clips yielded real words, neither m1 did; agrees with Whisper era-recall 9%->17% and record 56% clip at true 62k |
| BLIND #4 re-listen of B (m2) | — | "the third remembered any box larger" | Upgrade to 3/8 (37.5%): "remembered" exact (longest blind-recovered word ever, 4 syllables); "any"~"a heavy", "larger"~"last winter" phonetic echoes; m2 blind total ~25% |
| m3_long 6k (A100, true step 6k, c0 ~2.9) | c0 2.9 | USER: "really good"; Whisper: "the quick brown fox, just a lazy dog" WER 33% RECALL 67% | PROJECT RECORD by far (prev 56%). First sub-100% WER on fox probe. 6/9 words exact at only 6k steps into m3 — doubled corpus paying off fast |

## M3 BREAKTHROUGH — novel sentences intelligible (ckpt_6000, 2026-06-13 ~01:30)
Cooled (t0.6 k50) batch from m3 ckpt_6000, Whisper word-recall:
| ref | heard | recall |
| fox x3 | varies: "ground fox gents" 22%; "cron-flots lilies" 11%; "quick brown fox jumps from a lazy dog" 78% | hi-variance |
| She sells seashells by the seashore | "She sells seals by the seashore" | 83% |
| My brother bought a new red car | "My brother bought a new cart" | 71% |
| Please close the door before you leave | "Please close the door for relief" | 57% |
| The weather today is cold and windy | "The work today is all we would do" | 43% |
| The children played in the park all afternoon | degenerate ramble "thank you very much" | 0% |
Mean 46% (all 8); novel-sentence mean ~63% excl. degenerate. FIRST intelligible novel speech. Remaining flaws: per-sample variance, occasional 20s degeneration (exposure-bias drift). Validates train-longer+more-data thesis decisively; RQ2 (adapters teach frozen LLM speech) tipping to YES.
CORRECTION: m3test_07 is SILENT (peak 0.012 vs ~0.5 normal), not a ramble — model failed to initiate speech, ran to 20s cap emitting silence; Whisper's "thank you very much" was a silence-hallucination. So the failure mode is occasional SILENT DROPOUT, not noise-drift. Novel-sentence intelligibility is 4/4 on clips that actually spoke (83/71/57/43%). Fix candidates: min-length guard, greedy first frame, EOS-timing check.
SILENT-DROPOUT FIX VERIFIED: generate_wav (decode + re-sample on rms<0.01) — the children sentence that went silent now 3/3 voiced (rms 0.06), and intelligible: "The children play in a park all the time" 75% recall. So 5/5 novel sentences intelligible once dropout removed. Fix: gemma_csm/model.py generate_wav, used by generate.py + train.py save_samples.
COOLED 6k vs 16k head-to-head (fair test): 16k > 6k decisively. Fox mean recall 78% (16k: 67/89/78) vs 59% (6k: 67/67/44); seashells 50% vs 33%. New record clip ckpt16000 cooled "The quick brown fox jumps over the Louisa dog" 89% (8/9). User heard hot-flipbook 16k (single t0.9 sample, unlucky) vs hot 6k (lucky) — variance, not regression. FIX: save_samples now cooled (t0.6 k50) so flipbook clips are representative.
18k cooled test: best fox clip 100% recall ("The quick brown fox jumps over the lazy dog", WER 11%, tiny dog-dog stutter) — FIRST 100% on a NON-memorized sentence (vs the overfit 100% which was memorized = generalization milestone). Best-clip trend 6k 67% -> 16k 89% -> 18k 100%. 3-sample means noisy (59/78/70%); seashells unlucky 17% this round. Ceiling clearly rising.

## KNOWN ISSUES (EOS/length control family — distinct from intelligibility, which is solved)
1. **Trailing word echo**: model finishes sentence correctly then re-emits the last word before stopping (e.g. cmp18000_02 "...lazy dog-dog"). Cause: weak/late EOS firing; c0 EOS class under-trained vs content tokens (depth_weight 0.01 pushed training toward semantics). Cosmetic — words are correct. Fixes: (a) sharpens with more training; (b) gen-side trailing-repeat trim; (c) small EOS-prob boost near expected length.
2. **Silent dropout** (FIXED via generate_wav retry, but root cause remains): occasionally fails to initiate speech, runs to max_frames silent. Retry masks it; same EOS/initiation-timing root as #1.
Both are length/EOS-control issues, NOT content/intelligibility. Polish-phase work, post-M3.

## Long-sentence probe (18k cooled, max 250 frames)
Content intelligible for the OPENING clause (~8-12s) then degrades/mis-stops:
- "weather this morning cold and grey, but by the afternoon..." -> near-verbatim 75%, then ran to 20s cap (tail drift)
- "She walked slowly down the quiet street..." -> first clause perfect, ran to cap, 47%
- "My brother...drove to the coast...harbor..." -> captured but compressed, stopped early 5.4s, 67%
Mean 63% (carried by strong openings). DIAGNOSIS: intelligibility not the limit; SUSTAINING past one clause is. Two causes: (1) EOS/length-control (logged); (2) DATA — training clips capped ~13s (max_seconds 13, max_frames 160), so model never learned to sustain long utterances. Future lever: raise clip-length cap in data prep + EOS fixes.

CORRECTION (user ear): long clips don't drift into mush — they go SILENT after the opening clause. Unifies the bugs: silent-dropout (silence from frame 0) and long-clip tail-silence are the SAME failure — c0 collapses into a silent attractor, striking at frame 0 OR mid-clip (~after the 13s training horizon). Mid-clip case evades the generate_wav RMS retry (overall energy looks fine with a good opening + silent tail). Root: model never trained on >13s sequences, collapses past that horizon. Fixes: (cheap/cosmetic) detect silence-onset mid-clip and trim; (real) re-prep data at 20-25s cap so it learns to sustain.

CORRECTION 2 (measured energy envelope, user ear was right): long clips have only 2-5s of ACTUAL speech then silence — long_01 speech ends 2.5s (12% voiced), long_00 ends 4.6s (18% voiced). Earlier "8-12s opening" was WRONG (I inferred length from Whisper's word count; model actually CRAMS the words into 2-3s fast speech then collapses). Revised diagnosis: model emits only a few seconds of speech REGARDLESS of target length, then collapses to silence. Collapse point ~2-5s likely = MEAN training-clip length (most LibriTTS utts are short), not the 13s cap. So short sentences (fox ~3s) complete fine; long sentences get a rushed fragment + silence. Lesson: trust the audio/ear over Whisper word-count for length. Real fix still = longer+more-uniform-length training data + EOS/duration control.
20k cooled battery (parallel w/ encode): MEAN RECALL 66% (best battery yet). "Please close the door before you leave" 100% recall / 0% WER — cleanest generalization result, fully novel, no stutter (better than 18k fox 100% which had dog-dog). brother 86%, seashells 67%, fox x3 ~48%. INSIGHT: fox pangram is unusually hard (weird word seq); normal novel sentences score 67-100%, so fox-anchored metrics UNDERSTATE the model. Trend: generalization climbing steadily through m3.

## A100 FINAL (m3 ckpt_22000, training stopped at user request): step 22000, train c0 2.536, val c0 2.645
22k cooled battery MEAN RECALL 71% (best yet). "Please close the door" 100% (consistent w/ 20k). "The children played in the park all after[noon]" 88% — THE SAME SENTENCE THAT COLLAPSED TO SILENCE AT 6K now works (model got robust). fox x2 78%/78%, seashells 67%, brother 14% (unlucky). Cooled-battery trend: 6k ~46% -> 20k 66% -> 22k 71%. ckpt_22000 verified on HF. A100 phase done.

---
M4 FINAL — 48k battery (ckpt_40000, TB step 48k; cooled t0.6 k50, 10 clips, 5 reused + 5 novel). PROJECT RECORD: mean WER 13%, mean recall 88%.
- Novel set (91% recall / 9% WER) BEAT reused probes (84% / 17%) -> generalization, not memorization.
- "Please close the door before you leave" 100%/0%; novel 30-word "parking lot" paragraph 97%/3% near-verbatim; "little dog" 93%; fox 89% ("lazy goat").
- All 10 clips sustained full length (long ones 9.8-12.2s, 86-90% voiced) -> silence-collapse fully cured.
- Every remaining error is ACOUSTIC (goat/dog, Backdown/backbone, war-or-light/warm-light, snulled-fably-of-lines/smelled-faintly-of-lavender) not word-dropout -> M5 (acoustic rebalance) is the correct next lever.
- m4 c0 trend (all-5 reused comparability): TB 30k 75% -> 36k 79% -> 42k 71% -> 44k 81% -> 48k 84%; plateaued. Train c0 ~1.8-2.1, no held-out val tracked for m4 (m4b relaunch omitted --val-shard; restored for m5). Trainer stopped at ckpt_40000 for M5 handoff.

---
M5 first perceptual reading (acoustic rebalance, depth_weight 0.01->0.3, ckpt_4000, ~4k steps): PRIMED listen, Gianni reports clips sound "a little less robotic" than m4. CALIBRATION: primed (knew which was m5), small-n, early (4k). Consistent with theory: val_depth flat (val2000 6.565 -> val4000 6.554) because ~6.3 is largely the Mimi codec entropy floor, so the rebalance payoff shows up as timbre, NOT a depth-loss drop. Training c0 1.4-2.3 (no intelligibility regression). NOT yet confirmed -> blind A/B (m4-40000 vs m5-latest, shuffled) deferred to a later checkpoint (12k/20k) when the gap should be wider. m5 left running uninterrupted.

---
M5 FINAL — acoustic rebalance (depth_weight 0.01->0.3) = NEGATIVE RESULT. Stopped at ckpt_10000.
WER battery: m4-final + m5 {2,4,6,8,10}k x 5 sentences (30 clips), cooled. mean recall 80%, WER 26%.
- depth_acoustic FLAT 2k->10k: val depth 6.565/6.554/6.604/6.604/6.548. 30x weight bump moved acoustic loss ZERO. 2nd independent confirmation of the Mimi codec entropy floor (1st = capacity run: 150M==425M both pinned ~6.7).
- No intelligibility win: m5 recall bounces 74-89% (8k best, 10k regresses to m4 ~79%); within noise of m4.
- INTRODUCED stutter (depth_weight shock destabilized generation, invisible to CE loss = repeated valid tokens): s4_attic@4k repeats whole opening phrase (WER 64% recall 42%); mild repeats fox@6k ("the love of the"), printing@8k ("from all from all"), weather@8k ("but by but by"). Scattered, not a clean transient, not resolved by 10k. Confirms Gianni ear-report of 4k/6k stutter.
- VERDICT: loss-weight knob is a dead end for naturalness (no acoustic gain, no intelligibility gain, adds instability). Naturalness needs the flow-matching/diffusion acoustic head (continuous features, below the discrete-codec floor). m5 not resumed. GPU free.

---
UNSLOTH LOADER-SWAP VALIDATED on E2B (2026-06-14). scripts/verify_unsloth_loader.py: ALL PASS.
- Unsloth FastModel patches Gemma4 + get_peft_model LoRA; extracted text decoder has hidden_size=1536, get_input_embeddings, get_per_layer_inputs (PLE survives wrapper = the key unknown). forward(output_hidden_states + our c0/depth heads + PLE) + backward both work.
- Fix needed: _extract_text_model path base_model.model.model.language_model (Unsloth/PEFT wrapper); plain path returned multimodal model whose top config lacks hidden_size.
- VRAM: E2B via Unsloth 10.97GB (batch1 fwd+bwd) vs 12.2GB our default loader -> ~1.2GB lighter, with Unsloth cpp-ext DISABLED (torch 2.10<2.11). Good for 12B-fits-16GB.
- rev.18 loader-swap is real. Next: download gemma-4-12b, run same probe for real VRAM (blocked: ~25GB vs ~18GB free, needs disk cleanup). venv ~/unsloth-venv isolated from training .venv.

---
12B FITS (2026-06-14): google/gemma-4-12B-it probed via DEFAULT bitsandbytes loader (transformers 5.10.2 supports gemma4_unified; Unsloth 5.5.0 does NOT -> Unsloth-12B parked). Loads, heads attach, fwd+bwd+grads OK. PEAK VRAM 10.59GB (probe tiny-seq) -> ~5GB headroom; est ~12-14GB at real 250-frame training. LIGHTER than E2B (12.2GB) because 12B Unified is a clean dense decoder, NO PLE (hidden_size_per_layer_input=0), NO KV-share. Overturns rev.20 edge-case worry. RQ1 12B run feasible on free substrate, no cloud. Next build: audio-IN path (keep+freeze the native audio tower, scatter its embeds into inputs_embeds before our speech-out heads; cheap passthrough probe first; data is the hard part). See paper rev.23.

---
AUDIO-IN WIRING CONFIRMED (2026-06-14, scripts/probe_audio_in.py). gemma4_unified audio API: get_audio_features(input_features (N,640), mask) on full.model (Gemma4UnifiedModel) -> backbone-dim 3840; merge via masked_scatter at audio_token_id=258881. Probe (synthetic 640-d feats): PASS, (1,50,640)->(1,50,3840), 7.74GB. RQ1 audio-IN de-risked at wiring level. ONE blocker for real audio: Gemma4UnifiedProcessor (waveform->features) not importable in transformers 5.10.2 -> defer real-audio (build/source/wait for the feature extractor). Remaining build: audio-IN loader (keep full model), forward scatter, feature frontend, paired prosody data. See paper rev.24.

---
RQ1 GATES CLOSED (2026-06-14). Three blockers for the 12B thesis run resolved:
1. AUDIO-IN FORWARD validated: GemmaCSM.forward additive [text|audio-in|audio-out] branch (consumes precomputed audio_in_embeds, shifts output positions, text->audio path byte-identical when absent). Probe on 12B PASS: text->audio loss 18.5 (regression ok), audio-IN loss 18.1, grads flow 2.15e5, peak 11.05GB (batch1, ~4.5GB headroom). scripts/probe_audio_in_forward.py.
2. PROCESSOR located: Gemma4UnifiedProcessor absent in transformers 5.10.2, PRESENT in 5.13.0.dev0 (git source); needs "vision" backend (pillow+torchvision) to instantiate (unified img+audio+video processor). Audio-IN venv = transformers-from-source + torch/torchvision/pillow + bnb + audio deps.
3. DATASET chosen: CASIA-LM/OpenS2S_Datasets (HF, Apache-2.0, ungated, ~50k EN input-speech->emotion-conditioned response-speech pairs, synthetic prosody = fine for bootstrapping). Supplements CREMA-D (ODbL parallel emotion), AMI (CC-BY real prosody). On-task-but-NC: Expresso/Seamless/ESD/DeepDialogue.
Remaining = BUILD not unknowns: data pipeline (OpenS2S->input_features->get_audio_features->audio_in_embeds; output->Mimi codes), wire audio-encode into loop, audio-IN venv, accept heads train from scratch (d 1536->3840). See paper rev.25.

---
AUDIO-IN BUILD STARTED (2026-06-14). loader.py::load_gemma_csm_audio_in written (keeps full model for get_audio_features, LoRA text decoder, frozen audio tower, returns audio_encode). data_prep_opens2s.py written vs real manifest schema (query/response audio+text+emotion). OpenS2S EN wavs = 87.9GB compressed (query 19 + response 69) >> 29GB free -> data_prep must STREAM a subset from tars (no full download), encode on the fly. audio-IN venv ~/audioin-venv building: torch+torchvision cu128 + transformers-from-source(5.13) + bnb + peft + pillow + soundfile. Loader+forward validated on synthetic (rev.24/25); remaining: finish venv, stream-subset rewrite, few-pair test (verify processor audio key + Mimi resample), wire audio-encode into train.py. Paper rev.26.

---
OPENS2S DATA PIPELINE WORKS (2026-06-14). data_prep_opens2s.py --stream validated end-to-end on real data (audio-IN venv). Streaming-subset: tars not index-ordered -> take first-K encountered + intersect; 600/600 first-K -> 590 paired (98% overlap), leading bytes only (no 88GB DL). Real shard (14 enc/6 skip>20s): {input_features (138,640) fp16, codes (32,214) i64, text, query_text, emotion="Frustrated"}. Verify-points resolved: processor key=input_features, Mimi resample OK. Data side of RQ1 done. Remaining: training-loop wiring (OpenS2S dataset+collate, loop runs audio_encode->audio_in_embeds->forward). Loader+forward already validated. Paper rev.27.

---
AUDIO-IN TRAIN LOOP VALIDATED end-to-end (2026-06-14). dataset_opens2s.py (OpenS2SDataset + collate_opens2s w/ max_audio/max_frames caps) + scripts/probe_audio_in_train.py. Tiny test (real OpenS2S shard, batch1, audio48/codes100): PASS. text(1,54)+codes(1,32,101)+input_features(1,48,640) -> audio_encode -> audio_in_embeds(1,48,3840) -> forward loss 18.49 -> backward grads 1.11e5, peak 13.72GB. FULL RQ1 chain runs on real data on free 16GB card. QLoRA: yes (4bit NF4 + LoRA r32 all-linear); use PagedAdamW8bit for the real 12B run (memory). Memory: full real seq ~380tok OOMs batch1 (~15.5GB); capped ~200tok fits 13.72GB -> real run = short clips + batch1 + grad-accum + paged opt. Heads train from scratch (d3840). Every RQ1 component validated; remaining = train-loop assembly + shards + time, no unknowns. Paper rev.28.
