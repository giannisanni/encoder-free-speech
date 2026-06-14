"""Verify the Unsloth loader-swap (paper rev. 18) on E2B before committing to
the 12B download. Checks the one open unknown: does Unsloth tolerate
output_hidden_states=True + our custom heads + per_layer_inputs (PLE)
passthrough on the extracted text decoder?

Run (in a venv with unsloth installed):
    python -m scripts.verify_unsloth_loader
Needs ~8GB free VRAM (E2B via Unsloth). Run when m5 has freed the GPU.
"""
import torch

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm, load_gemma_csm_unsloth


def ok(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}",
          flush=True)
    return cond


def main():
    cfg = GemmaCSMConfig()
    import os
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    print("backbone:", cfg.backbone_id, flush=True)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
    print(f"loading {cfg.backbone_id} via Unsloth FastModel ...", flush=True)
    import os as _os
    which = _os.environ.get("LOADER", "unsloth")
    print("loader:", which, flush=True)
    model, tok = (load_gemma_csm(cfg) if which == "default"
                  else load_gemma_csm_unsloth(cfg))
    bb = model.backbone

    all_ok = True
    all_ok &= ok("backbone.config.hidden_size present",
                 hasattr(bb, "config") and hasattr(bb.config, "hidden_size"),
                 f"hidden_size={getattr(getattr(bb,'config',None),'hidden_size','?')}")
    all_ok &= ok("get_input_embeddings()", hasattr(bb, "get_input_embeddings"))
    all_ok &= ok("PLE: get_per_layer_inputs present (model.uses_ple)",
                 model.uses_ple,
                 "rev.18 KEY UNKNOWN -- if FAIL, PLE was stripped by Unsloth wrapper")

    # synthetic batch matching collate() output
    B, Tt, Tf = 1, 8, 10
    dev = "cuda"
    batch = {
        "text_ids": torch.randint(1, 100, (B, Tt), device=dev),
        "text_mask": torch.ones(B, Tt, dtype=torch.bool, device=dev),
        "codes": torch.randint(0, 2048, (B, 32, Tf), device=dev),
        "frame_mask": torch.ones(B, Tf + 1, dtype=torch.bool, device=dev),
    }
    # codes need Tf+1 width? collate pads to fmax = max_frames+1; frame_mask is
    # Tf+1 wide. Match: widen codes to Tf+1 too.
    batch["codes"] = torch.randint(0, 2048, (B, 32, Tf + 1), device=dev)

    try:
        out = model(batch)
        loss = float(out["loss"])
        all_ok &= ok("forward() runs (output_hidden_states + custom heads + PLE)",
                     loss == loss, f"loss={loss:.3f} c0={float(out['loss_c0']):.3f} "
                     f"depth={float(out['loss_depth']):.3f}")
    except Exception as e:
        all_ok &= ok("forward() runs", False, f"{type(e).__name__}: {e}")

    try:
        out["loss"].backward()
        g = sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.requires_grad and p.grad is not None)
        all_ok &= ok("backward() + grads flow to trainables", g > 0, f"sum|grad|={g:.2e}")
    except Exception as e:
        all_ok &= ok("backward()", False, f"{type(e).__name__}: {e}")

    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    print(f"\n{'=== ALL CHECKS PASSED ===' if all_ok else '=== SOME CHECKS FAILED ==='}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
