import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.model import GemmaCSM


def tiny_backbone():
    # Llama stands in for the Gemma4 text stack on CPU: same causal-LM
    # interface (inputs_embeds, output_hidden_states), which is all the
    # wrapper uses. This tests OUR wiring, not Google's weights.
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=512)
    return LlamaForCausalLM(cfg)


def make_model():
    cfg = GemmaCSMConfig(depth_d_model=64, depth_layers=1, depth_heads=2)
    return GemmaCSM(cfg, backbone=tiny_backbone()), cfg


def test_forward_loss():
    model, cfg = make_model()
    batch = {
        "text_ids": torch.randint(0, 500, (2, 7)),
        "text_mask": torch.ones(2, 7, dtype=torch.bool),
        "codes": torch.randint(0, 2048, (2, 32, 10)),
        "frame_mask": torch.ones(2, 10, dtype=torch.bool),
    }
    out = model(batch)
    assert out["loss_c0"].item() > 0 and out["loss_depth"].item() > 0


def test_generate_smoke():
    model, cfg = make_model()
    codes = model.generate_codes(torch.randint(0, 500, (1, 5)), max_frames=4)
    assert codes.shape[0] == 32 and 1 <= codes.shape[1] <= 4
