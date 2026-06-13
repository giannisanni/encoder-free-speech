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
