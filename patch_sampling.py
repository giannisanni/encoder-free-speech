"""Upgrade train_audio_in.py:
  1. checkpoints save {model, opt, step}  (was flat trainables)
  2. --resume restores model + optimizer + start_step  (clean resumes hereafter)
  3. --sample-every N: decode a few clips in-process at each eval, reusing the
     resident model (no 2nd GPU copy). Guarded so a sampling OOM never kills training.
"""
import re
f = "gemma_csm/train_audio_in.py"
s = open(f).read()
orig = s

# -- ensure `import re` present -------------------------------------------------
if "\nimport re\n" not in s and not s.startswith("import re\n"):
    s = s.replace("import time\n", "import time\nimport re\n", 1)

# -- A. sampling args -----------------------------------------------------------
A_anchor = "    a = p.parse_args()\n"
A_new = (
    '    p.add_argument("--sample-every", type=int, default=0,\n'
    '                   help="every N steps decode a few clips in-process (0=off)")\n'
    '    p.add_argument("--sample-clips", type=int, default=3)\n'
    '    p.add_argument("--sample-frames", type=int, default=150)\n'
    "    a = p.parse_args()\n"
)
assert A_anchor in s, "A anchor"
s = s.replace(A_anchor, A_new, 1)

# -- B. resume block: format-aware, capture start_step + opt state --------------
B_old = '''    if a.resume:
        sd = torch.load(a.resume, weights_only=True, map_location="cpu")
        own = model.state_dict()
        for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
            del sd[k]
        model.load_state_dict(sd, strict=False)
        del sd, own
        torch.cuda.empty_cache()
        print(f"resumed trainables from {a.resume}", flush=True)
'''
B_new = '''    start_step = 0
    resume_opt = None
    if a.resume:
        ckpt = torch.load(a.resume, weights_only=True, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and "step" in ckpt:
            sd, resume_opt, start_step = ckpt["model"], ckpt.get("opt"), int(ckpt["step"])
        else:
            sd = ckpt
            mm = re.search(r"(\\d+)", Path(a.resume).stem)
            start_step = int(mm.group(1)) if mm else 0
        own = model.state_dict()
        for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
            del sd[k]
        model.load_state_dict(sd, strict=False)
        del sd, own
        torch.cuda.empty_cache()
        print(f"resumed from {a.resume} at step {start_step} "
              f"(optimizer {'restored' if resume_opt is not None else 'fresh'})", flush=True)
'''
assert B_old in s, "B anchor"
s = s.replace(B_old, B_new, 1)

# -- C. restore optimizer after it is built ------------------------------------
C_old = '''    opt = make_paged_opt(model, cfg.lr, a.depth_lr_mult)
    a.out.mkdir(parents=True, exist_ok=True)
'''
C_new = '''    opt = make_paged_opt(model, cfg.lr, a.depth_lr_mult)
    if resume_opt is not None:
        try:
            opt.load_state_dict(resume_opt)
            print("optimizer state restored", flush=True)
        except Exception as e:
            print(f"WARN optimizer restore failed ({e}); continuing fresh", flush=True)
    del resume_opt
    a.out.mkdir(parents=True, exist_ok=True)
'''
assert C_old in s, "C anchor"
s = s.replace(C_old, C_new, 1)

# -- D. checkpoint save format --------------------------------------------------
D_old = '        torch.save(trainable_state_dict(model), a.out / f"ckpt_{step}.pt")\n'
D_new = ('        torch.save({"model": trainable_state_dict(model),\n'
         '                    "opt": opt.state_dict(), "step": step},\n'
         '                   a.out / f"ckpt_{step}.pt")\n')
assert D_old in s, "D anchor"
s = s.replace(D_old, D_new, 1)

# -- E + F. do_sample def, then start step at start_step -----------------------
EF_old = '''        print(f"saved ckpt_{step}", flush=True)

    step, t0 = 0, time.time()
'''
EF_new = '''        print(f"saved ckpt_{step}", flush=True)

    _codec = {}
    def do_sample(step):
        if not a.sample_every or step % a.sample_every != 0:
            return
        import torchaudio
        from gemma_csm.codec import MimiCodec
        if "c" not in _codec:
            _codec["c"] = MimiCodec(device="cuda")
        codec = _codec["c"]
        sdir = a.out / "samples"; sdir.mkdir(parents=True, exist_ok=True)
        pool = [it.get("text", "").strip() for it in (val_items or ds.items)]
        prompts = [t for t in pool if 20 <= len(t) <= 160][:a.sample_clips]
        model.eval(); torch.cuda.empty_cache()
        try:
            for i, text in enumerate(prompts):
                ids = torch.tensor([tok(text)["input_ids"]], device="cuda")
                wav, codes, att = model.generate_wav(
                    ids, codec, max_frames=a.sample_frames, temperature=0.9, top_k=50)
                torchaudio.save(str(sdir / f"step{step:06d}_s{i}.wav"),
                                wav.detach().cpu().unsqueeze(0), 24000)
                print(f"sample step{step} s{i} {wav.numel()/24000:.1f}s att={att}", flush=True)
        except RuntimeError as e:
            print(f"WARN sampling skipped (likely VRAM): {e}", flush=True)
        finally:
            model.train(); torch.cuda.empty_cache()

    step, t0 = start_step, time.time()
'''
assert EF_old in s, "EF anchor"
s = s.replace(EF_old, EF_new, 1)

# -- G. call do_sample at eval boundary ----------------------------------------
G_old = '''            if step % a.eval_every == 0 or step >= a.steps:
                save_ckpt(step)
'''
G_new = '''            if step % a.eval_every == 0 or step >= a.steps:
                save_ckpt(step)
                do_sample(step)
'''
assert G_old in s, "G anchor"
s = s.replace(G_old, G_new, 1)

# -- throughput uses steps-this-session ----------------------------------------
s = s.replace("/ max(step, 1):.2f}s/step", "/ max(step - start_step, 1):.2f}s/step", 1)

assert s != orig, "no change made"
open(f, "w").write(s)
import ast
ast.parse(s)
print("PATCH OK")
print("  sample-every arg:", "--sample-every" in s)
print("  start_step resume:", "start_step = int(ckpt" in s)
print("  opt restore:", "opt.load_state_dict(resume_opt)" in s)
print("  ckpt dict save:", '"opt": opt.state_dict()' in s)
print("  do_sample call:", "do_sample(step)" in s)
