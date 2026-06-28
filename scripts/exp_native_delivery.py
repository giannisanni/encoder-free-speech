"""Delivery-fix experiment (unified model): does routing the query audio through
Gemma-4's NATIVE multimodal forward (input_ids with <audio> placeholders +
input_features -> the model's own masked_scatter) carry lexical CONTENT into the
reply-position hidden states -- the states a speech head would condition on?

Why this exists
---------------
The current `generate_unified` (model.py) SPLICES projected audio embeds into the
EXTRACTED language_model: `self.backbone(inputs_embeds=[pre | ain | suf])`. A
zeroed-audio ablation proved that path BYPASSES the audio (identical greedy output
with real vs zeroed audio). This script tests the hypothesis that the bug is
purely DELIVERY: same framing tokens, same `input_features`, but route through the
FULL `Gemma4UnifiedForConditionalGeneration` so the model does its native audio
handling (masked_scatter + per-layer embeddings) instead of a hand-built splice.

Decisive pass/fail
------------------
  (A) reply text from native delivery CHANGES when we zero the audio, and the
      real-audio reply is on-topic for the query  -> native delivery carries
      content (splice did not). GREEN: build head-on-native-hidden-states.
  (B) reply-position last-hidden-states differ (norm delta) real vs zeroed
      -> the states a head would read DEPEND on the audio. This is the signal
      that a head conditioned on them can be grounded.
If both fail, the problem is deeper than delivery and we learn that for ~1 pod-hr.

It also INTROSPECTS the full-model forward signature / audio kwargs / token ids,
which is exactly what the model.py `forward_unified` rewrite needs.

Run (on a GPU pod with the audio-IN venv + a held-out OpenS2S shard):
  GEMMA_BACKBONE=google/gemma-4-12B-it \
  .venv/bin/python -m scripts.exp_native_delivery --shard /path/to/heldout.pt --index 0
"""
import argparse
import inspect
import os

import torch

from gemma_csm.config import GemmaCSMConfig

# Exact framing generate_unified uses (model.py): <bos> user\n <start_of_audio> ...
# <audio>*N ... <end_of_audio> <end_of_turn>\n model\n <channel>thought<end_channel>
_UNI_PREFIX = (2, 105, 2364, 107, 256000)
_UNI_SUFFIX = (258883, 106, 107, 105, 4368, 107, 100, 45518, 107, 101)


def load_full(cfg):
    """Load the FULL multimodal Gemma in the hybrid (audio path bf16) quant, the
    way loader.load_gemma_csm_audio_in does -- but keep `full` (for .generate)
    and `mm` (owns get_audio_features). No LoRA: we probe the frozen native path."""
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    try:
        proc = AutoProcessor.from_pretrained(cfg.backbone_id)
    except Exception as e:  # processor optional; we reconstruct from shard features
        print("note: AutoProcessor unavailable:", e, flush=True)
        proc = None
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["audio_tower", "embed_audio", "vision_tower",
                               "embed_vision", "multi_modal_projector",
                               "embedding_projection", "lm_head", "embed_tokens"],
    )
    full = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation=os.environ.get("GEMMA_ATTN", "sdpa"))
    mm = getattr(full, "model", full)
    assert hasattr(mm, "get_audio_features"), "not a unified multimodal backbone"
    return full, mm, tok, proc


def audio_kwarg_names(full):
    """Map our (features, mask) onto whatever this model's forward actually names
    them -- the contract the forward_unified rewrite must follow."""
    sig = set(inspect.signature(full.forward).parameters)
    print("forward params:", sorted(sig), flush=True)
    feat = next((k for k in ("input_features", "audio_values", "input_audio_embeds")
                 if k in sig), None)
    mask = next((k for k in ("input_features_mask", "audio_values_mask",
                             "input_audio_embeds_mask", "audio_attention_mask")
                 if k in sig), None)
    print(f"--> audio feature kwarg = {feat!r}  mask kwarg = {mask!r}", flush=True)
    return feat, mask


def native_inputs(tok, mm, feats, fmask, dev):
    """Build the NATIVE delivery inputs: input_ids = framing + <audio>*N placeholders,
    where N MUST equal the audio-token count the model's own get_audio_features
    produces (so its internal masked_scatter aligns)."""
    atok = mm.config.audio_token_id if hasattr(mm.config, "audio_token_id") else 258881
    with torch.no_grad():
        try:
            af = mm.get_audio_features(feats, fmask, return_dict=True)
            af = af.pooler_output if hasattr(af, "pooler_output") else af
        except TypeError:
            af = mm.get_audio_features(feats)
    af = af if isinstance(af, torch.Tensor) else af[0]
    n = af.shape[-2] if af.dim() == 3 else af.shape[0]
    ids = list(_UNI_PREFIX) + [atok] * n + list(_UNI_SUFFIX)
    print(f"audio tokens N={n} | input_ids len={len(ids)} | audio_token_id={atok}", flush=True)
    return torch.tensor([ids], device=dev), n


@torch.no_grad()
def gen_native(full, input_ids, feat_kw, feats, mask_kw, fmask, max_new):
    kw = {"input_ids": input_ids,
          "attention_mask": torch.ones_like(input_ids),
          "max_new_tokens": max_new, "do_sample": False}
    if feat_kw:
        kw[feat_kw] = feats
    if mask_kw and fmask is not None:
        kw[mask_kw] = fmask
    out = full.generate(**kw)
    return out[0, input_ids.shape[1]:]


@torch.no_grad()
def teacher_forced(full, input_ids, reply_ids, feat_kw, feats, mask_kw, fmask):
    """Teacher-force the reply; return the LAST hidden state AND the logits at the
    positions that predict the reply tokens -- the states a speech head consumes."""
    full_ids = torch.cat([input_ids, reply_ids[None].to(input_ids.device)], dim=1)
    kw = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids),
          "output_hidden_states": True}
    if feat_kw:
        kw[feat_kw] = feats
    if mask_kw and fmask is not None:
        kw[mask_kw] = fmask
    out = full(**kw)
    sl = slice(input_ids.shape[1] - 1, -1)                  # positions predicting reply
    return out.hidden_states[-1][:, sl].float(), out.logits[:, sl].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="held-out OpenS2S .pt shard")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()

    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    dev = "cuda"
    full, mm, tok, proc = load_full(cfg)
    feat_kw, mask_kw = audio_kwarg_names(full)

    items = torch.load(a.shard, weights_only=True)
    it = items[a.index]
    feats = it["input_features"][None].to(dev, torch.bfloat16)        # (1, Ta, 640)
    fmask = torch.ones(feats.shape[:2], dtype=torch.bool, device=dev)
    tgt_reply = it.get("text", "")
    transcript = it.get("query_text", "")
    print(f"\n=== item {a.index} ===\nquery transcript: {transcript!r}\ntarget reply: {tgt_reply!r}\n", flush=True)

    input_ids, n = native_inputs(tok, mm, feats, fmask, dev)
    zeros = torch.zeros_like(feats)

    # (A) native delivery: real vs zeroed audio -> does the reply depend on audio?
    real = tok.decode(gen_native(full, input_ids, feat_kw, feats, mask_kw, fmask, a.max_new), skip_special_tokens=True).strip()
    zero = tok.decode(gen_native(full, input_ids, feat_kw, zeros, mask_kw, fmask, a.max_new), skip_special_tokens=True).strip()
    print(f"\n[NATIVE]   real-audio reply: {real!r}", flush=True)
    print(f"[NATIVE] zeroed-audio reply: {zero!r}", flush=True)
    changed = real != zero
    print(f"\n[{'PASS' if changed else 'FAIL'}] (A) native reply CHANGES with audio "
          f"(splice did NOT) -> delivery {'carries' if changed else 'still drops'} content", flush=True)

    # (B) reply-position hidden states: do the head-input states depend on audio?
    # NORM is a bad proxy (dominated by the identical teacher-forced reply tokens);
    # measure DIRECTION (cosine distance) + behavioural divergence (do the next-token
    # predictions differ?) -- those are what a head conditioned on the states sees.
    reply_ids = torch.tensor(tok(tgt_reply)["input_ids"], device=dev)
    hr, lr = teacher_forced(full, input_ids, reply_ids, feat_kw, feats, mask_kw, fmask)
    hz, lz = teacher_forced(full, input_ids, reply_ids, feat_kw, zeros, mask_kw, fmask)
    cos_dist = (1 - torch.nn.functional.cosine_similarity(hr, hz, dim=-1)).mean().item()
    l2_rel = ((hr - hz).norm(dim=-1) / hr.norm(dim=-1).clamp_min(1e-6)).mean().item()
    disagree = (lr.argmax(-1) != lz.argmax(-1)).float().mean().item()
    print(f"\nreply-hidden: cosine-dist={cos_dist:.3f}  rel-L2={l2_rel:.3%}  "
          f"next-token disagreement real-vs-zeroed={disagree:.1%}", flush=True)
    grounded = cos_dist > 0.05 or disagree > 0.10
    print(f"[{'PASS' if grounded else 'FAIL'}] (B) head-input hidden states DEPEND on audio "
          f"(cosine/behaviour) -> a head conditioned on them can be grounded", flush=True)

    print("\npeak VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB", flush=True)
    print("\n=== VERDICT ===", flush=True)
    if changed and grounded:
        print("DELIVERY WAS THE BUG. Native forward carries content into reply hidden", flush=True)
        print("states. Next: rewrite forward_unified to run the FULL model and feed the", flush=True)
        print("CSM head from output_hidden_states[-1] at the reply positions.", flush=True)
    else:
        print("Native delivery did NOT ground the reply. The problem is deeper than", flush=True)
        print("the splice -- investigate placeholder alignment / framing / PLE before", flush=True)
        print("committing to the head-on-hidden-states rewrite.", flush=True)


if __name__ == "__main__":
    main()
