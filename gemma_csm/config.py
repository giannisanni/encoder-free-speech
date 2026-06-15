from dataclasses import dataclass


@dataclass
class GemmaCSMConfig:
    backbone_id: str = "google/gemma-4-E2B-it"
    mimi_id: str = "kyutai/mimi"

    # Mimi codec geometry
    n_codebooks: int = 32          # set 8 for fast iteration (codes are precomputed for all 32)
    codebook_size: int = 2048
    frame_rate: float = 12.5
    sample_rate: int = 24_000

    # c0 head: codebook vocab + 1 EOS class
    c0_eos_id: int = 2048

    # depth decoder
    depth_d_model: int = 1024
    depth_layers: int = 4
    depth_heads: int = 8
    depth_loss_frame_frac: float = 1 / 16  # CSM memory trick
    # Moshi weights semantic:acoustic 100:1; implemented by scaling depth DOWN
    # so c0 gradient magnitudes (and tuned lr) stay unchanged.
    depth_loss_weight: float = 1.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # QLoRA paper: adapter coverage (all linear layers) is the critical
    # variable, not rank; attention-only LoRA is the documented-failing
    # config for new-modality token generation (arXiv 2401.00246).
    lora_all_linear: bool = False

    # training
    lr: float = 3e-4
    batch_size: int = 2
    grad_accum: int = 8
    max_frames: int = 160          # ~12.8 s clips max for M0
