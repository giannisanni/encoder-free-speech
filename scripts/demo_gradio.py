"""Interactive Gradio demo: talk to gemma-csm, two modes with a toggle.

  * Cascade  : native frozen-backbone understanding (audio -> reply text, pure
               backbone, adapters off) -> CSM speech head voices it. Reliable.
  * Unified  : one forward -- generate_unified delivers audio natively and the head
               voices the backbone's latent reply. Research result; OOD-flaky.

One backbone is loaded; the head + adapters are swapped per mode (fits 16 GB).
Mic capture needs a secure context, so we launch with share=True (HTTPS *.gradio.live
link). For a private alternative: launch without share and `ssh -L 7860:localhost:7860`.

Run (substrate): ~/audioin-venv/bin/python -m scripts.demo_gradio
"""
import os
import tempfile
import wave

import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in

DEVICE = "cuda"
CKPT_REPO = "giannisan/gemma-csm-ckpts"

cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
print("loading backbone (hybrid)...", flush=True)
model, tok, _ = load_gemma_csm_audio_in(cfg, device=DEVICE)
for m in (getattr(model, "full_model", None), model.backbone):
    try:
        m.gradient_checkpointing_disable()
    except Exception:
        pass
proc = AutoProcessor.from_pretrained(cfg.backbone_id)
codec = MimiCodec(device=DEVICE)

# --- LoRA strength control: scale the adapter 0..1 (0 = pristine frozen backbone,
# 1 = full adapter). LoRA layers are shared between backbone and full_model, so
# scaling them here affects both seg1 and seg2 of generate_unified. ---
_LORA_BASE = {}


def _index_lora():
    for name, mod in model.backbone.named_modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict):
            _LORA_BASE[name] = dict(sc)


def set_lora_strength(s):
    s = float(s)
    for name, mod in model.backbone.named_modules():
        base = _LORA_BASE.get(name)
        if base:
            for adp, b in base.items():
                if adp in mod.scaling:
                    mod.scaling[adp] = b * s


_index_lora()
print(f"indexed {len(_LORA_BASE)} LoRA layers for strength control", flush=True)


def _load_head(path):
    p = hf_hub_download(CKPT_REPO, path, repo_type="model")
    c = torch.load(p, map_location="cpu", weights_only=True)
    return c["model"] if isinstance(c, dict) and "model" in c else c


print("loading head checkpoints...", flush=True)
HEAD_SD = {"Unified": _load_head("unified_a6000/ckpt_3000.pt"),
           "Cascade": _load_head("head_v2/ckpt_8000.pt")}
_cur = {"mode": None}


def _ensure_head(mode):
    if _cur["mode"] != mode:
        model.load_state_dict(HEAD_SD[mode], strict=False)
        _cur["mode"] = mode


def _save_wav(path, wav, sr=24000):
    x = (wav.detach().cpu().float().clamp(-1, 1).numpy().reshape(-1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())


def _mic_to_float(audio):
    sr, data = audio
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")
    if np.abs(data).max() > 1.5:        # int16 PCM -> [-1,1]
        data = data / 32768.0
    return sr, data


@torch.no_grad()
def respond(audio, mode, lora_strength):
    if audio is None:
        return "(record a query first)", None
    sr, data = _mic_to_float(audio)
    if data.size < sr * 0.2:
        return "(too short -- hold the button and speak)", None
    _ensure_head(mode)

    if mode == "Unified":
        set_lora_strength(lora_strength)   # slider: 0 = pristine reasoning, 1 = full adapter
        feat = proc(audio=data, sampling_rate=sr, return_tensors="pt")
        inf = feat["input_features"].squeeze(0).to(DEVICE, torch.bfloat16)
        text, codes = model.generate_unified(inf, tok, tok.eos_token_id, max_text=96,
                                              max_frames=400, temperature=0.6, top_k=50,
                                              text_temperature=0.6)
        wav = codec.decode(codes)
    else:  # Cascade: native understanding (pure backbone) -> reply text -> speak
        set_lora_strength(1.0)   # head_v2 LoRA full for speaking; Stage A disables it anyway
        tin = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _save_wav(tin, torch.from_numpy(data), sr=sr)
        msgs = [{"role": "user", "content": [
            {"type": "audio", "path": tin},
            {"type": "text", "text": "Please respond to what I said."}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")
        dev = model.full_model.device
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
        n_in = inp["input_ids"].shape[1]
        with model.backbone.disable_adapter():
            out = model.full_model.generate(**inp, max_new_tokens=96, do_sample=False)
        text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
        ids = torch.tensor([tok(f"[lj] {text}", add_special_tokens=True)["input_ids"]], device=DEVICE)
        wav, _codes, _att = model.generate_wav(ids, codec, max_frames=400,
                                               temperature=0.6, top_k=50)

    outp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    _save_wav(outp, wav)
    return text, outp


with gr.Blocks(title="Talk to gemma-csm") as demo:
    gr.Markdown(
        "# Talk to gemma-csm\n"
        "Speak a query; the **frozen** Gemma-4 12B backbone replies in speech.\n\n"
        "- **Cascade** -- native understanding then the speech head voices the reply "
        "(reliable; the 91.9%-recall system).\n"
        "- **Unified** -- one forward pass, the head voices the backbone's latent reply "
        "(the research result; great on training-style prompts, drifts to templates "
        "out-of-distribution).")
    mode = gr.Radio(["Cascade", "Unified"], value="Cascade", label="Mode")
    strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                         label="Unified LoRA strength  (1.0 = full adapter / template-collapse, "
                               "0.0 = pristine frozen-backbone reasoning; try ~0.3-0.5). "
                               "Only affects Unified mode.")
    inp = gr.Audio(sources=["microphone"], type="numpy", label="Your spoken query")
    btn = gr.Button("Send", variant="primary")
    txt = gr.Textbox(label="Reply (text the model decided)")
    out = gr.Audio(label="Spoken reply", autoplay=True)
    btn.click(respond, [inp, mode, strength], [txt, out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
