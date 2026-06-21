# encoder-free-speech

Conversational speech generation on a **frozen, encoder-free multimodal LLM**.

This project couples a frozen [Gemma-4](https://ai.google.dev/gemma) "Unified" backbone
(Google's encoder-free, natively-multimodal variant) with a
[CSM](https://www.sesame.com/)-style speech-output head: a semantic-codebook
predictor plus a depth transformer over [Mimi](https://github.com/kyutai-labs/moshi)
residual-vector-quantized codes. The backbone stays frozen; only QLoRA adapters and
the speech head are trained, on a single 16 GB consumer GPU. The bet: a backbone that
already *understands* audio can generate speech conditioned on understood context,
without bolting on a separate audio encoder.

This repository accompanies the paper *Conversational Speech Generation on an
Encoder-Free Multimodal Backbone* (`paper/main.pdf`).

## What's here

A reproducible pipeline plus several transferable findings:

- **A frozen, quantized LLM made to speak.** Adapters + the speech head alone reach
  **13% WER / 88% word recall** on held-out text-to-speech, on 16 GB.
- **Conversational speech-to-speech with no backbone training**, via a two-stage
  cascade: native-pathway understanding into the frozen backbone, then the speech
  head voices the reply.
- **4-bit NF4 quantization silently destroys the backbone's audio comprehension**
  while sparing prosody. Keeping the audio tower + projector in bf16 (a ~10.5 GB
  "hybrid" load) restores verbatim understanding. See `gemma_csm/loader.py`
  (`llm_int8_skip_modules`).
- **Delivery matters:** the backbone's native multimodal pathway conveys lexical
  content; a hand-built feature-splice of the same features conveys prosody but not
  content.
- **Gate on free-running WER, not teacher-forced loss.** Teacher-forced codebook
  cross-entropy (c0) is not a proxy for intelligibility (a memorizing model scores
  well yet decodes to 141% WER).

## Architecture

```
                 frozen Gemma-4 12B (hybrid: 4-bit text / bf16 audio)
 audio in  --->  native multimodal pathway  --->  hidden state  --->  c0 head (semantic)
 (16 kHz)                                                        \--> depth transformer (31 acoustic)
                                                                        |
                                                              frozen Mimi decoder ---> speech out
```

Only the QLoRA adapters, the audio-frame embeddings, the c0 head, and the depth
transformer are trained. See Figure 1 (target single model) and Figure 2 (the
demonstrated cascade) in the paper.

## Install

Requires an NVIDIA GPU (16 GB+ for the 12B). The audio-IN path needs transformers
built from source (for `Gemma4UnifiedProcessor`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U "git+https://github.com/huggingface/transformers.git" \
  bitsandbytes peft accelerate datasets soundfile pillow numpy \
  tqdm tensorboard jiwer faster-whisper whisper-normalizer "huggingface_hub[cli]"
# verify the encoder-free backbone is available:
python -c "from transformers.models import gemma4_unified; print('ok')"
```

You must accept Google's [Gemma Terms](https://ai.google.dev/gemma/terms) to download
`google/gemma-4-12B-it`.

## Data

Phase A (text->speech) uses LibriTTS / LJSpeech; Phase B (audio-in) uses the English
split of [OpenS2S](https://huggingface.co/datasets/CASIA-LM/OpenS2S_Datasets). The
OpenS2S audio (~88 GB) is streamed, not downloaded, and Mimi-encoded into shards:

```bash
# stream a subset of OpenS2S query/response pairs into training shards
python -m gemma_csm.data_prep_opens2s --stream --n-pairs 2000 --out data/shards_opens2s
# longer-reply subset (>= N words):
python -m gemma_csm.data_prep_opens2s --stream --candidates 2200 --min-words 50 \
  --n-pairs 600 --out data/shards_opens2s_long
```

## Train

Single 16 GB GPU, batch 1 + grad-accum 16, QLoRA r32 on all linear layers:

```bash
python -m gemma_csm.train_audio_in \
  --shard "data/shards_opens2s/*.pt" \
  --out checkpoints/run1 --steps 8000 --eval-every 2000 --val-every 1000 \
  --max-frames 256 --max-audio 128 --max-text 96 \
  --batch-size 1 --grad-accum 16 --num-workers 0 \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --depth-frac 0.25 --depth-weight 0.02 --grad-clip 1.0 --lr 5e-5 \
  --backbone-id google/gemma-4-12B-it
```

`--resume <ckpt>` restores model, step, and optimizer. On 16 GB use
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Evaluate

Always gate on free-running decode, not teacher-forced loss. Decode clips from a
checkpoint, then score word error rate against the references with faster-whisper:

```bash
# 1. decode free-running clips from a checkpoint
python decode_tts.py --ckpt checkpoints/run1/ckpt_8000.pt \
  --shards "data/shards_opens2s/*.pt" --n-clips 20 --out samples_eval
# 2. score WER over a manifest of (decoded audio, reference text)
python -m gemma_csm.eval_wer --manifest wer_manifest.tsv
```

## Checkpoints and shards

Released on Hugging Face (subject to the licenses in `NOTICE`):

- Checkpoints: [`giannisan/gemma-csm-ckpts`](https://huggingface.co/giannisan/gemma-csm-ckpts)
- Data shards: [`giannisan/gemma-csm-shards`](https://huggingface.co/datasets/giannisan/gemma-csm-shards)

## Status

The two-stage cascade is the demonstrated conversational system. A single end-to-end
model (audio delivered through the native pathway inside one network, head retrained
on those hidden states) is in progress. See the paper's discussion for the roadmap.

## License

Code in this repository is licensed under **Apache-2.0** (see `LICENSE`). This does
**not** relicense the models, datasets, or checkpoints it depends on or produces;
see `NOTICE` for their terms (Gemma-4 under Google's Gemma Terms, Mimi under Kyutai's
terms, LibriTTS CC BY 4.0, OpenS2S Apache-2.0; released checkpoints are Gemma
derivatives and inherit the Gemma Terms).

## Citation

```bibtex
@misc{sanrochman2026gemmacsm,
  title  = {Conversational Speech Generation on an Encoder-Free Multimodal Backbone},
  author = {Sanrochman, Gianni},
  year   = {2026},
  note   = {https://github.com/giannisanni/encoder-free-speech}
}
```
