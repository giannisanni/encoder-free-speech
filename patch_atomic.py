"""Make checkpoint saves atomic: write ckpt_N.pt.tmp then os.replace -> ckpt_N.pt.
A pkill mid-save can then only truncate the .tmp, never a real checkpoint."""
f = "gemma_csm/train_audio_in.py"
s = open(f).read()

if "\nimport os\n" not in s and not s.startswith("import os"):
    s = s.replace("import time\n", "import os\nimport time\n", 1)

old = '''    def save_ckpt(step):
        torch.save({"model": trainable_state_dict(model),
                    "opt": opt.state_dict(), "step": step},
                   a.out / f"ckpt_{step}.pt")'''
new = '''    def save_ckpt(step):
        dst = a.out / f"ckpt_{step}.pt"
        tmp = a.out / f"ckpt_{step}.pt.tmp"
        torch.save({"model": trainable_state_dict(model),
                    "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, dst)'''
assert old in s, "save_ckpt anchor not found"
s = s.replace(old, new, 1)

open(f, "w").write(s)
import ast
ast.parse(s)
print("ATOMIC SAVE PATCHED:", "os.replace(tmp, dst)" in s, "| import os:", "import os" in s)
