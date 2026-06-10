import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def load_gemma_csm(cfg: GemmaCSMConfig, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    # text-only path for M0: if the checkpoint is the multimodal wrapper,
    # unwrap to the language model (we feed inputs_embeds directly)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
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
