# Gemma-CSM

A Sesame-CSM-style speech generation model with Google's Gemma 4 (encoder-free,
natively multimodal) as the backbone instead of a text-only Llama. The bet:
a backbone that already *understands* audio and video can eventually generate
speech conditioned on understood multimodal context, which CSM and MisoTTS
architecturally cannot.

- Design: `docs/plans/2026-06-10-gemma-csm-design.md`
- Implementation plan (M0): `docs/plans/2026-06-10-gemma-csm-m0-implementation.md`

Dev happens on the Mac; training runs on `substrate` (RTX 4060 Ti 16 GB) —
sync with `scripts/sync-to-substrate.sh`. CPU tests: `pytest`. GPU tests
(substrate): `pytest -m gpu`.
