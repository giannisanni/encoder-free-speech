import torch

# --- set_submodule shim ---
# transformers' bnb 4-bit path calls nn.Module.set_submodule(), which only exists
# in torch >= 2.5. Some GPU host images (e.g. the RunPod PyTorch 2.4 template) ship
# an older torch, so we backfill the method when missing. No-op on torch >= 2.5.
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target, module):
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = _set_submodule
# --- end shim ---

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def _extract_text_model(full):
    """Pull the text decoder out of Gemma 4's multimodal wrapper.

    M0 is text->audio only, so we drop the vision/audio towers (saves VRAM).
    The text decoder keeps its PLE embedding tables.
    """
    for path in ("base_model.model.model.language_model",
                 "base_model.model.language_model",
                 "model.model.language_model",
                 "model.language_model", "language_model"):
        obj = full
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return full  # already a bare causal LM


def load_gemma_csm(cfg: GemmaCSMConfig, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        # KEEP the audio/vision towers + projector + tied lm_head in bf16. NF4 on the
        # audio projector destroys speech COMPREHENSION (text survives, audio doesn't) --
        # confirmed: 4-bit -> "I'm a text-based AI"; hybrid -> transcribes verbatim, ~same VRAM.
        # Passing a skip-list overrides the default auto-skip, so list lm_head/embed_tokens too.
        llm_int8_skip_modules=["audio_tower", "embed_audio", "vision_tower", "embed_vision",
                               "multi_modal_projector", "embedding_projection",
                               "lm_head", "embed_tokens"],
    )
    import os
    full = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation=os.environ.get("GEMMA_ATTN", "sdpa"),  # GEMMA_ATTN=eager to debug kernel OOB
    )
    backbone = _extract_text_model(full)
    if backbone is not full:
        del full
        torch.cuda.empty_cache()

    # Gemma 4 wraps projections in Gemma4ClippableLinear; peft can only wrap
    # the inner nn.Linear, so target those by regex (text decoder layers only).
    if cfg.lora_all_linear:
        targets = (r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                   r"|mlp\.(gate|up|down)_proj)(\.linear)?$")
    else:
        targets = r"layers\.\d+\.self_attn\.(q|k|v|o)_proj(\.linear)?$"
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=targets,
    )
    backbone = get_peft_model(backbone, lora)
    if os.environ.get("GEMMA_GC", "on") != "off":   # GEMMA_GC=off to debug checkpointing corruption
        backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
    model = GemmaCSM(cfg, backbone=backbone)
    # new modules in bf16 on GPU; quantized backbone manages its own placement
    model.audio_embs.to(device, torch.bfloat16)
    model.c0_head.to(device, torch.bfloat16)
    model.depth.to(device, torch.bfloat16)
    return model, tok


# --- Unsloth loader-swap (paper rev. 18) -----------------------------------
# Load the FROZEN Gemma backbone via Unsloth FastModel (real Gemma kernels +
# 4-bit + "unsloth" gradient checkpointing) and attach our UNCHANGED audio
# heads. The architecture (model.py) is untouched; only the backbone load
# swaps. This is the prototype that verifies rev. 18's one open unknown:
#   Unsloth must tolerate output_hidden_states=True + custom heads +
#   per_layer_inputs (PLE) passthrough on the extracted text decoder.
# Returns the same (model, tok) tuple as load_gemma_csm so train.py and
# generate.py can switch on a flag. Requires `unsloth` in the active venv.
def load_gemma_csm_unsloth(cfg: GemmaCSMConfig, device: str = "cuda",
                           max_seq_length: int = 2048):
    from unsloth import FastModel  # lazy: only imported when --unsloth is used

    # 1. Load the full multimodal Gemma 4 in 4-bit via Unsloth.
    full, tok = FastModel.from_pretrained(
        model_name=cfg.backbone_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        full_finetuning=False,
    )

    # 2. LoRA on the LANGUAGE layers only (freeze vision/audio towers -- per
    #    rev. 17, training Gemma's own audio layers needs >15GB; we use the
    #    frozen pretrained audio-in path for understanding). Unsloth applies
    #    LoRA to the full model and routes by the finetune_* flags, keeping us
    #    on its accelerated path including PLE handling.
    full = FastModel.get_peft_model(
        full,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=cfg.lora_all_linear,
        use_gradient_checkpointing="unsloth",
        target_modules=(["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
                        if cfg.lora_all_linear else
                        ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    # 3. Extract the text decoder (drops vision/audio towers from the forward,
    #    keeps PLE tables + LoRA adapters). _extract_text_model walks the (now
    #    peft/unsloth-wrapped) tree; if the wrapper changes the attribute path
    #    this is the first thing to adjust -- verify_unsloth_loader.py probes
    #    exactly that.
    backbone = _extract_text_model(full)

    # 4. Attach our unchanged heads (model.py as-is).
    model = GemmaCSM(cfg, backbone=backbone)
    model.audio_embs.to(device, torch.bfloat16)
    model.c0_head.to(device, torch.bfloat16)
    model.depth.to(device, torch.bfloat16)
    return model, tok


# --- Audio-IN loader (paper rev. 25, RQ1) ----------------------------------
# Load the full multimodal Gemma ONCE, keep it for get_audio_features (the
# frozen audio tower + projector = audio understanding), apply LoRA to its text
# decoder, and attach our speech-out heads. Returns (model, tok, audio_encode):
#   audio_encode(input_features, input_features_mask) -> (B, Ta, d) backbone-dim
# embeddings to drop into batch["audio_in_embeds"]. The full model is kept alive
# via the audio_encode closure (unlike load_gemma_csm, which deletes it).
def load_gemma_csm_audio_in(cfg: GemmaCSMConfig, device: str = "cuda"):
    import torch
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        # KEEP the audio/vision towers + projector + tied lm_head in bf16. NF4 on the
        # audio projector destroys speech COMPREHENSION (text survives, audio doesn't) --
        # confirmed: 4-bit -> "I'm a text-based AI"; hybrid -> transcribes verbatim, ~same VRAM.
        # Passing a skip-list overrides the default auto-skip, so list lm_head/embed_tokens too.
        llm_int8_skip_modules=["audio_tower", "embed_audio", "vision_tower", "embed_vision",
                               "multi_modal_projector", "embedding_projection",
                               "lm_head", "embed_tokens"],
    )
    import os
    full = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation=os.environ.get("GEMMA_ATTN", "sdpa"),  # GEMMA_ATTN=eager to debug kernel OOB
    )
    # inner Gemma4UnifiedModel owns get_audio_features (audio tower + projector)
    mm = getattr(full, "model", full)
    assert hasattr(mm, "get_audio_features"), \
        "backbone has no get_audio_features -- not a unified multimodal model?"

    # the text decoder (LoRA target); audio tower stays frozen
    backbone = _extract_text_model(full)
    cov = getattr(cfg, "lora_coverage", "") or ("all" if cfg.lora_all_linear else "attn")
    if cov == "all":
        targets = (r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                   r"|mlp\.(gate|up|down)_proj)(\.linear)?$")
    elif cov == "attn_down":   # lighter: attn + MLP down_proj only (less trunk perturbation)
        targets = (r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                   r"|mlp\.down_proj)(\.linear)?$")
    else:                      # attn-only (q,k,v,o)
        targets = r"layers\.\d+\.self_attn\.(q|k|v|o)_proj(\.linear)?$"
    print(f"LoRA coverage={cov} r={cfg.lora_r} alpha={cfg.lora_alpha} "
          f"(scaling {cfg.lora_alpha / cfg.lora_r:.3g})", flush=True)
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=targets,
    )
    backbone = get_peft_model(backbone, lora)
    if os.environ.get("GEMMA_GC", "on") != "off":   # GEMMA_GC=off to debug checkpointing corruption
        backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()

    model = GemmaCSM(cfg, backbone=backbone)
    model.audio_embs.to(device, torch.bfloat16)
    model.c0_head.to(device, torch.bfloat16)
    model.depth.to(device, torch.bfloat16)
    # Expose the FULL multimodal model for NATIVE audio delivery (forward_unified).
    # LoRA was injected in-place into full.language_model, so full(input_ids=..,
    # input_features=..) runs the LoRA-adapted text decoder through the real
    # multimodal integration (masked_scatter + token-type/modality attention) --
    # the path proven (scripts/exp_native_delivery) to carry lexical content, which
    # the hand-built splice into the extracted decoder does NOT. backbone (the same
    # underlying language_model, peft-wrapped) is reused for the audio-OUT frames
    # via KV continuation. audio_token_id identifies the <audio> placeholders.
    # Store via __dict__ to BYPASS nn.Module.__setattr__ submodule registration:
    # `self.full_model` shares weights with `backbone` (LoRA is injected in-place)
    # but registering it would make model.state_dict() save the whole 12B again
    # (observed: 9.6 GB checkpoints) and break resume. As a plain attribute it is
    # still reachable as self.full_model but excluded from state_dict/.parameters().
    model.__dict__["full_model"] = full
    model.audio_token_id = int(getattr(mm.config, "audio_token_id", 258881))

    @torch.no_grad()
    def audio_encode(input_features, input_features_mask=None):
        """input_features: (B, num_tokens, audio_embed_dim) -> (B, Ta, d_model).
        Frozen audio tower; run under no_grad (we don't train Gemma's audio path).
        """
        out = mm.get_audio_features(input_features, input_features_mask,
                                    return_dict=True)
        return out.pooler_output if hasattr(out, "pooler_output") else out

    return model, tok, audio_encode
