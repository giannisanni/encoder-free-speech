"""#1 Pure-backbone native baseline: feed spoken queries through the FROZEN Gemma's
own multimodal chat template + generate -- NO LoRA, NO speech head. If raw Gemma
answers OOD questions (identity, Gemma facts) from audio, the backbone's
intelligence is genuinely active (zero added training). Isolates the backbone.

Run (substrate): GEMMA_BACKBONE=google/gemma-4-12B-it \
  ~/audioin-venv/bin/python -m scripts.test_native_baseline --wav-dir ~/ood_wavs
"""
import argparse
import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MID = os.environ.get("GEMMA_BACKBONE", "google/gemma-4-12B-it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    a = ap.parse_args()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["audio_tower", "embed_audio", "vision_tower",
                               "embed_vision", "multi_modal_projector",
                               "embedding_projection", "lm_head", "embed_tokens"])
    model = AutoModelForCausalLM.from_pretrained(
        MID, quantization_config=bnb, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    proc = AutoProcessor.from_pretrained(MID)

    for wp in sorted(glob.glob(os.path.join(a.wav_dir, "*.wav"))):
        messages = [{"role": "user", "content": [
            {"type": "audio", "path": wp},
            {"type": "text", "text": "Please respond to what I said."}]}]
        try:
            inputs = proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        except Exception as e:
            print(f"[{os.path.basename(wp)}] apply_chat_template failed: {e}", flush=True)
            continue
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
        print(f"\n[{os.path.basename(wp)}] NATIVE-GEMMA reply: {text!r}", flush=True)
    print("\nNATIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
