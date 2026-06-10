import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma_csm.config import GemmaCSMConfig


def _causal_block(cfg: GemmaCSMConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.depth_d_model, nhead=cfg.depth_heads,
        dim_feedforward=cfg.depth_d_model * 4, batch_first=True,
        norm_first=True, dropout=0.0,
    )
    return nn.TransformerEncoder(layer, num_layers=cfg.depth_layers)


class DepthDecoder(nn.Module):
    """Per-frame: from backbone hidden h_t and c0, autoregressively predict c1..c31.

    c0 itself comes from the backbone's head, never from here. Sequence per
    frame: position k carries proj(h_t) + emb_k(c_k) + pos_k for k = 0..30,
    and position k's output predicts codebook k+1.
    """

    def __init__(self, cfg: GemmaCSMConfig, backbone_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.depth_d_model
        self.h_proj = nn.Linear(backbone_dim, d)
        # one embedding table per input codebook level 0..30
        self.embs = nn.ModuleList(
            [nn.Embedding(cfg.codebook_size, d) for _ in range(cfg.n_codebooks - 1)]
        )
        self.pos = nn.Parameter(torch.zeros(cfg.n_codebooks - 1, d))
        self.tf = _causal_block(cfg)
        # one output head per predicted level 1..31
        self.heads = nn.ModuleList(
            [nn.Linear(d, cfg.codebook_size) for _ in range(cfg.n_codebooks - 1)]
        )

    def _inputs(self, h, codes_in):
        # h: (N, backbone_dim); codes_in: (N, K) levels 0..K-1 -> (N, K, d)
        toks = torch.stack(
            [self.embs[k](codes_in[:, k]) for k in range(codes_in.shape[1])], dim=1
        )
        return self.h_proj(h).unsqueeze(1) + toks + self.pos[: codes_in.shape[1]]

    def loss(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """h: (N, backbone_dim), codes: (N, 32). Teacher-forced CE over c1..c31."""
        K = self.cfg.n_codebooks - 1
        x = self._inputs(h, codes[:, :K])
        mask = nn.Transformer.generate_square_subsequent_mask(K, device=x.device)
        y = self.tf(x, mask=mask, is_causal=True)              # (N, K, d)
        losses = [
            F.cross_entropy(self.heads[k](y[:, k]), codes[:, k + 1])
            for k in range(K)
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate(self, h: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        """h: (N, backbone_dim), c0: (N,) -> full frames (N, 32)."""
        K = self.cfg.n_codebooks - 1
        out = [c0]
        for k in range(K):
            x = self._inputs(h, torch.stack(out, dim=1))
            mask = nn.Transformer.generate_square_subsequent_mask(k + 1, device=x.device)
            y = self.tf(x, mask=mask, is_causal=True)
            out.append(self.heads[k](y[:, -1]).argmax(-1))
        return torch.stack(out, dim=1)


class GemmaCSM(nn.Module):
    """Backbone (any HF causal LM) + audio frame embeddings + c0 head + DepthDecoder.

    Sequence: [text tokens][audio frames...]. Audio frame t input embedding =
    sum over 32 codebook embeddings. Backbone hidden at the LAST TEXT position
    predicts frame 0's c0; hidden at frame t predicts frame t+1's c0 (shifted
    next-frame prediction, standard causal LM but in frame space). EOS class
    (id 2048) on the c0 head terminates generation.
    """

    def __init__(self, cfg: GemmaCSMConfig, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        # Gemma 4 PLE: passing inputs_embeds without input_ids triggers a
        # vocab-sized reverse lookup (OOM, and audio embeds aren't in vocab).
        # We must compute per_layer_inputs ourselves: real lookup for text
        # positions, zeros for audio frames.
        self.uses_ple = hasattr(backbone, "get_per_layer_inputs")
        d_model = backbone.config.hidden_size
        self.audio_embs = nn.ModuleList(
            [nn.Embedding(cfg.codebook_size, d_model) for _ in range(cfg.n_codebooks)]
        )
        self.c0_head = nn.Linear(d_model, cfg.codebook_size + 1)  # +1 = EOS
        self.depth = DepthDecoder(cfg, backbone_dim=d_model)

    # -- embedding assembly --------------------------------------------------
    def _embed_frames(self, codes):
        # codes: (B, 32, T) -> (B, T, d)
        return torch.stack(
            [self.audio_embs[k](codes[:, k]) for k in range(self.cfg.n_codebooks)]
        ).sum(0)

    def _embed_text(self, text_ids):
        return self.backbone.get_input_embeddings()(text_ids)

    def _backbone_kwargs(self, text_ids, n_audio_frames):
        if not self.uses_ple:
            return {}
        ple = self.backbone.get_per_layer_inputs(text_ids, None)  # (B, Tt, L, D)
        if n_audio_frames:
            pad = ple.new_zeros(ple.shape[0], n_audio_frames, *ple.shape[2:])
            ple = torch.cat([ple, pad], dim=1)
        return {"per_layer_inputs": ple}

    # -- training forward ------------------------------------------------------
    def forward(self, batch):
        """Packs each item contiguously as [text][audio frames] with ALL
        padding at the tail. Padding between text and audio would shift the
        audio positions of short-text items relative to inference (which has
        no padding), corrupting what the model learns about position geometry.
        """
        text_e = self._embed_text(batch["text_ids"])             # (B, Tt, d)
        audio_e = self._embed_frames(batch["codes"])             # (B, Tf, d)
        B, Tt = batch["text_ids"].shape
        Tf = batch["codes"].shape[2]
        L = batch["text_mask"].sum(1)                            # text lens
        Fr = batch["frame_mask"].sum(1) - 1                      # real frame counts

        x = text_e.new_zeros(B, Tt + Tf, text_e.shape[-1])
        attn = torch.zeros(B, Tt + Tf, dtype=torch.long, device=x.device)
        ple = None
        if self.uses_ple:
            ple_text = self.backbone.get_per_layer_inputs(batch["text_ids"], None)
            ple = ple_text.new_zeros(B, Tt + Tf, *ple_text.shape[2:])
        for i in range(B):
            li, fi = int(L[i]), int(Fr[i])
            x[i, :li] = text_e[i, :li]
            x[i, li : li + fi] = audio_e[i, :fi]
            attn[i, : li + fi] = 1
            if ple is not None:
                ple[i, :li] = ple_text[i, :li]
        kwargs = {"per_layer_inputs": ple} if ple is not None else {}
        h = self.backbone(
            inputs_embeds=x, attention_mask=attn, output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1]

        # c0: for item i, frame t (0..Fr_i incl. EOS) is predicted from
        # position L_i - 1 + t — identical to inference geometry.
        pred_hs, c0_tgts, depth_hs, depth_codes = [], [], [], []
        c0_logits_padded = h.new_zeros(B, Tf, self.cfg.codebook_size + 1)
        for i in range(B):
            li, fi = int(L[i]), int(Fr[i])
            pos = torch.arange(li - 1, li + fi, device=h.device)
            pred_hs.append(h[i, pos])                            # (fi+1, d)
            tgt = torch.cat([
                batch["codes"][i, 0, :fi],
                batch["codes"].new_tensor([self.cfg.c0_eos_id]),
            ])
            c0_tgts.append(tgt)
            depth_hs.append(h[i, pos[:-1]])                      # pre-frame hiddens
            depth_codes.append(batch["codes"][i, :, :fi].T)      # (fi, 32)
        pred_h = torch.cat(pred_hs)
        logits = self.c0_head(pred_h)
        loss_c0 = F.cross_entropy(logits.float(), torch.cat(c0_tgts))
        off = 0
        for i in range(B):
            n = int(Fr[i]) + 1
            c0_logits_padded[i, :n] = logits[off : off + n]
            off += n

        # depth loss on a random fraction of frames, conditioned on the
        # pre-frame hidden (no leakage, matches inference)
        all_h = torch.cat(depth_hs)
        all_codes = torch.cat(depth_codes)
        if len(all_h) == 0:
            loss_depth = loss_c0.new_zeros(())
        else:
            n_keep = max(1, int(len(all_h) * self.cfg.depth_loss_frame_frac))
            sel = torch.randperm(len(all_h))[:n_keep]
            loss_depth = self.depth.loss(all_h[sel], all_codes[sel])

        return {"loss_c0": loss_c0, "loss_depth": loss_depth,
                "loss": loss_c0 + loss_depth, "c0_logits": c0_logits_padded}

    # -- inference ---------------------------------------------------------------
    @torch.no_grad()
    def generate_codes(self, text_ids, max_frames: int = 1000,
                       temperature: float = 0.9, top_k: int = 0):
        """text_ids: (1, Tt) -> (32, T) codes. No KV cache; fine for M0 eval."""
        x = self._embed_text(text_ids)                            # (1, Tt, d)
        frames = []
        for _ in range(max_frames):
            h = self.backbone(
                inputs_embeds=x, output_hidden_states=True,
                **self._backbone_kwargs(text_ids, len(frames)),
            ).hidden_states[-1][:, -1]                            # (1, d)
            logits = self.c0_head(h).float() / temperature
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)                    # (1, 32)
            frames.append(frame[0])
            x = torch.cat([x, self._embed_frames(frame.unsqueeze(-1))], dim=1)
        if not frames:
            return torch.zeros(32, 1, dtype=torch.long)
        return torch.stack(frames, dim=1)                         # (32, T)
