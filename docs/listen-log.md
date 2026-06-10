# Listening log — perceptual evaluations by Gianni (the project's primary eval instrument)

Novel-sentence probe is s1: "The quick brown fox jumps over the lazy dog" (not in training data).

| Run / step | depth loss | Gianni's description | Interpretation |
|---|---|---|---|
| M0a v1, 1k | ~3.4 (leaky) | "scrambled, like a robot trying to produce speech", "speech played backwards on old tape recorders" | F1 leakage: semantic stream fine, acoustic garbage at inference |
| M0a v2, 1k | n/a | "a gasp", 0.2 s | F2 padding geometry: premature EOS |
| M0a v3, 4k | ~0.02 | full first clause "clear, sounded good", tail garbled ~0.1-0.2 s | memorization works; mild exposure-bias drift at tail |
| M0b, 8k | ~7.0 | "garbled mumbling, but I think I heard the last word fox if I listen closely" (position mismatch -> treated as pareidolia) | semantic stream generalizing, acoustic stream near-random |
| M0b, 22k | ~6.1 | "still mumbling but sounds better than the 8000 file; a few sounds that could potentially sound like the words if I listen closely" | perceptual improvement tracks depth-loss decline; proto-words emerging |
| M0b, 38k (~final) | ~6.5 | "I kinda could make up 'the quick brown fox' but that is only because I know what it should say; still not understandable speech" | PRIMED intelligibility reached on novel text; blind intelligibility not. Baseline ends acoustic-bottlenecked, as the flat depth loss predicted |

Pattern so far: c0 (semantic) loss governs speech-shaped rhythm and duration; depth (acoustic) loss governs word crispness. Perceptual reports have tracked depth loss monotonically.
