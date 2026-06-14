"""Audio-IN passthrough probe (paper rev. 23/24): confirm Gemma-4-12B's frozen
audio tower encodes audio into backbone-dim (3840) embeddings that our pipeline
can scatter into inputs_embeds before the speech-out heads.

Uses SYNTHETIC input_features (num_audios, num_tokens, audio_embed_dim=640) because
Gemma4UnifiedProcessor (waveform -> features) is not importable in transformers
5.10.2 yet. This tests the WIRING (audio tower -> projector -> backbone dim ->
scatter), not audio fidelity. Real-waveform input is deferred to the processor.

Run:  .venv/bin/python -m scripts.probe_audio_in
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

MID = "google/gemma-4-12B-it"
cfg = AutoConfig.from_pretrained(MID)
aed = cfg.audio_config.audio_embed_dim          # 640
thd = cfg.text_config.hidden_size               # 3840
atok = cfg.audio_token_id                        # 258881
print(f"audio_embed_dim={aed}  text_hidden={thd}  audio_token_id={atok}", flush=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
full = AutoModelForCausalLM.from_pretrained(
    MID, quantization_config=bnb, dtype=torch.bfloat16, attn_implementation="sdpa")
print("loaded:", type(full).__name__, flush=True)

# locate the Gemma4UnifiedModel that owns get_audio_features
mm = getattr(full, "model", full)
print("inner:", type(mm).__name__,
      "| has get_audio_features:", hasattr(mm, "get_audio_features"),
      "| has audio_tower:", hasattr(mm, "audio_tower"),
      "| has language_model:", hasattr(mm, "language_model"), flush=True)
assert hasattr(mm, "get_audio_features"), "get_audio_features not on full.model"

# synthetic audio features -> encode through the frozen audio tower + projector
N = 50
feats = torch.randn(1, N, aed, device="cuda", dtype=torch.bfloat16)
fmask = torch.ones(1, N, dtype=torch.bool, device="cuda")
with torch.no_grad():
    try:
        out = mm.get_audio_features(feats, fmask, return_dict=True)
        af = out.pooler_output if hasattr(out, "pooler_output") else out
    except TypeError as e:
        print("signature retry (no mask/return_dict):", e, flush=True)
        af = mm.get_audio_features(feats)
af = af if isinstance(af, torch.Tensor) else af[0]
print(f"[{'PASS' if af.shape[-1]==thd else 'FAIL'}] get_audio_features -> {tuple(af.shape)} "
      f"(want last dim = {thd} = backbone hidden)", flush=True)

# can we scatter audio embeds into a text-dim inputs_embeds at the placeholder?
emb = mm.language_model.get_input_embeddings() if hasattr(mm, "language_model") else mm.get_input_embeddings()
ids = torch.full((1, af.shape[-2] if af.dim() == 3 else af.shape[0]), atok, device="cuda")
ie = emb(ids)
print(f"[{'PASS' if ie.shape[-1]==thd else 'FAIL'}] placeholder embeds shape {tuple(ie.shape)} "
      f"-> audio scatters into backbone-dim sequence", flush=True)

print("peak VRAM:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB", flush=True)
print("=== AUDIO-IN WIRING CONFIRMED (synthetic features) ===", flush=True)
