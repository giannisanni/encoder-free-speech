"""Quick demo server for gemma-csm (run on substrate, reach via ssh -L tunnel).

Loads the frozen Gemma-4 12B + ckpt_20000 speech head + Mimi + Gemma audio
processor ONCE, then serves a one-page UI with two modes:
  * /tts      text  -> the model's voice           (Phase-A capability, works)
  * /converse mic   -> encoder-free audio-in -> speech out  (WIRING TEST: the
              model was not trained to respond to audio yet, so the reply will
              not be coherent -- this exercises the pipeline end to end.)

Launch:  ~/audioin-venv/bin/uvicorn serve_demo:app --host 127.0.0.1 --port 8000
Reach :  ssh -L 8000:localhost:8000 gianni@substrate  ->  http://localhost:8000
"""
import io
import os
import subprocess
import wave

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

from gemma_csm.codec import MimiCodec
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.model import GemmaCSMConfig

DEVICE = "cuda"
print(f"loading model + ckpt {os.environ.get('DEMO_CKPT', 'phaseB_opens2s/ckpt_32000.pt')} ...", flush=True)
cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha = 32, 16
cfg.lora_all_linear = True
model, tok, audio_encode = load_gemma_csm_audio_in(cfg, device=DEVICE)

ckpt = hf_hub_download(os.environ.get("DEMO_CKPT_REPO", "giannisan/gemma-csm-ckpts"),
                       os.environ.get("DEMO_CKPT", "phaseB_opens2s/ckpt_32000.pt"),
                       repo_type="model")
sd = torch.load(ckpt, weights_only=True, map_location="cpu")
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
own = model.state_dict()
for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
    del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
codec = MimiCodec(device=DEVICE)
proc = AutoProcessor.from_pretrained(cfg.backbone_id)
print("READY", flush=True)


def wav_bytes(wav, sr=24000):
    x = wav.detach().cpu().float().clamp(-1, 1).numpy().reshape(-1)
    pcm = (x * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_audio_16k(raw: bytes):
    """Any browser container (webm/opus, ogg, wav) -> 16 kHz mono float32."""
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=raw, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def audio_in_embeds_from_pcm(pcm, sr=16000):
    feat = proc(audio=pcm, sampling_rate=sr, return_tensors="pt")
    inf = None
    for k in ("input_features", "audio_values", "input_audio_embeds"):
        if k in feat:
            inf = feat[k]
            break
    if inf is None:
        raise KeyError(f"no audio features; processor keys: {list(feat.keys())}")
    mask = feat.get("input_features_mask")
    ain = audio_encode(inf.to(DEVICE, torch.bfloat16),
                       mask.to(DEVICE) if mask is not None else None)
    return ain if torch.is_tensor(ain) else ain.pooler_output


app = FastAPI()


@app.get("/")
def index():
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html")
    try:
        with open(p) as f:
            return HTMLResponse(f.read())   # read fresh each request -> edit demo.html, just refresh
    except FileNotFoundError:
        return HTMLResponse(HTML)            # embedded fallback


@app.post("/tts")
def tts(text: str = Form(...)):
    prompt = f"[lj] {text.strip()}"
    ids = torch.tensor([tok(prompt, add_special_tokens=True)["input_ids"]], device=DEVICE)
    wav, codes, att = model.generate_wav(ids, codec, max_frames=250,
                                         temperature=0.7, top_k=50)
    return Response(wav_bytes(wav), media_type="audio/wav",
                    headers={"X-Frames": str(codes.shape[1]), "X-Attempts": str(att)})


@app.post("/converse")
async def converse(audio: UploadFile = File(...)):
    raw = await audio.read()
    pcm = decode_audio_16k(raw)
    if pcm.size < 1600:  # < 0.1 s
        return Response(b"", media_type="audio/wav", headers={"X-Error": "too short"})
    ain = audio_in_embeds_from_pcm(pcm, 16000)
    ids = torch.tensor([tok("[lj] ", add_special_tokens=True)["input_ids"]], device=DEVICE)
    # min_frames forces ~2.5s of output (ban early EOS): ckpt_20000 isn't trained
    # to respond to audio so it would otherwise stop at 1 frame -- this makes the
    # pipeline audible (gibberish, but you hear voice-in -> voice-out work).
    # cap at 48 frames (~3.8s): the untrained audio-in model has no clean stop and
    # rambles to ~170 frames otherwise. It's gibberish anyway, so keep it short+fast.
    wav, codes, att = model.generate_wav(ids, codec, max_frames=48,
                                         temperature=0.8, top_k=50,
                                         audio_in_embeds=ain, min_frames=24, retries=1)
    return Response(wav_bytes(wav), media_type="audio/wav",
                    headers={"X-Frames": str(codes.shape[1]), "X-Attempts": str(att),
                             "X-InSeconds": f"{pcm.size/16000:.1f}"})


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>gemma-csm demo</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 16px;color:#1d1d1f}
 h1{font-size:20px} h2{font-size:15px;margin-top:28px}
 .card{border:1px solid #e0e0e0;border-radius:12px;padding:16px;margin:14px 0}
 textarea{width:100%;height:64px;font-size:15px;border:1px solid #ccc;border-radius:8px;padding:8px;box-sizing:border-box}
 button{font-size:15px;padding:9px 16px;border:0;border-radius:8px;background:#0071e3;color:#fff;cursor:pointer;margin-top:8px}
 button:disabled{background:#aaa;cursor:default}
 #rec.recording{background:#d70015}
 .note{color:#86868b;font-size:12.5px;line-height:1.5}
 audio{width:100%;margin-top:10px}
 .status{font-size:12.5px;color:#0071e3;margin-top:6px;min-height:16px}
</style></head><body>
<h1>gemma-csm &mdash; ckpt_20000</h1>

<div class="card">
 <h2>Make it speak (text &rarr; voice)</h2>
 <textarea id="txt">The old house stood silent on the hill.</textarea>
 <button id="say">Speak</button>
 <div class="status" id="s1"></div>
 <audio id="a1" controls></audio>
</div>

<div class="card">
 <h2>Talk to it (mic &rarr; voice)</h2>
 <p class="note">Wiring test: this checkpoint was trained to speak, not to respond to
 audio, so the reply will run end-to-end but won't be a coherent answer yet.</p>
 <button id="rec">&#127908; Hold to record</button>
 <div class="status" id="s2"></div>
 <audio id="a2" controls></audio>
</div>

<script>
const $=id=>document.getElementById(id);
$('say').onclick=async()=>{
 $('say').disabled=true;$('s1').textContent='generating...';
 const fd=new FormData();fd.append('text',$('txt').value);
 const r=await fetch('/tts',{method:'POST',body:fd});
 const b=await r.blob();$('a1').src=URL.createObjectURL(b);$('a1').play();
 $('s1').textContent='done ('+ (r.headers.get('X-Frames')||'?') +' frames, att '+(r.headers.get('X-Attempts')||'?')+')';
 $('say').disabled=false;
};
let mr,chunks;
async function start(){
 const st=await navigator.mediaDevices.getUserMedia({audio:true});
 mr=new MediaRecorder(st);chunks=[];
 mr.ondataavailable=e=>chunks.push(e.data);
 mr.onstop=async()=>{
  $('s2').textContent='thinking...';
  const blob=new Blob(chunks,{type:mr.mimeType});
  const fd=new FormData();fd.append('audio',blob,'rec.webm');
  const r=await fetch('/converse',{method:'POST',body:fd});
  if(r.headers.get('X-Error')){$('s2').textContent='error: '+r.headers.get('X-Error');return;}
  const b=await r.blob();$('a2').src=URL.createObjectURL(b);$('a2').play();
  $('s2').textContent='heard '+(r.headers.get('X-InSeconds')||'?')+'s in, '+(r.headers.get('X-Frames')||'?')+' frames out';
 };
 mr.start();$('rec').classList.add('recording');$('rec').textContent='&#9209; Release to send';
}
function stop(){if(mr&&mr.state!=='inactive'){mr.stop();mr.stream.getTracks().forEach(t=>t.stop());}$('rec').classList.remove('recording');$('rec').innerHTML='&#127908; Hold to record';}
const rec=$('rec');
rec.onmousedown=start;rec.onmouseup=stop;rec.onmouseleave=stop;
rec.ontouchstart=e=>{e.preventDefault();start();};rec.ontouchend=e=>{e.preventDefault();stop();};
</script>
</body></html>"""
