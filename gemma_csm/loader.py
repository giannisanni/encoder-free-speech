import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def _extract_text_model(full):
    """Pull the text decoder out of Gemma 4's multimodal wrapper.

    M0 is text->audio only, so we drop the vision/audio towers (saves VRAM).
    The text decoder keeps its PLE embedding tables.
    """
    for path in ("model.language_model", "language_model"):
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
    )
    full = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    backbone = _extract_text_model(full)
    if backbone is not full:
        del full
        torch.cuda.empty_cache()

    # Gemma 4 wraps projections in Gemma4ClippableLinear; peft can only wrap
    # the inner nn.Linear, so target those by regex (text decoder layers only).
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=r"layers\.\d+\.self_attn\.(q|k|v|o)_proj(\.linear)?$",
    )
    backbone = get_peft_model(backbone, lora)
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
    model = GemmaCSM(cfg, backbone=backbone)
    # new modules in bf16 on GPU; quantized backbone manages its own placement
    model.audio_embs.to(device, torch.bfloat16)
    model.c0_head.to(device, torch.bfloat16)
    model.depth.to(device, torch.bfloat16)
    return model, tok
