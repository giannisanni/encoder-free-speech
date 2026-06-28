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
        embs = [self.audio_embs[k](codes[:, k]) for k in range(self.cfg.n_codebooks)]
        out = torch.stack(embs).sum(0)
        if not torch.isfinite(out).all():
            wbad = [k for k in range(self.cfg.n_codebooks)
                    if not torch.isfinite(self.audio_embs[k].weight).all()]
            ebad = [k for k in range(self.cfg.n_codebooks) if not torch.isfinite(embs[k]).all()]
            print(f"_embed_frames NaN: codes dtype={codes.dtype} range=[{int(codes.min())},"
                  f"{int(codes.max())}] | NaN-weight cbs={wbad} | NaN-emb cbs={ebad}", flush=True)
        return out

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
        """Packs each item contiguously as [text][audio-in?][audio frames] with
        ALL padding at the tail. The optional audio-in segment -- precomputed
        backbone-dim embeddings from Gemma's frozen audio tower in
        batch["audio_in_embeds"] (B, Ta, d) with per-item lengths in
        batch["audio_in_mask"] -- sits between text and the speech-output frames
        so the decoder conditions its spoken reply on HOW the input was said
        (RQ1). When absent (Ta=0) the geometry is identical to text->audio.
        """
        text_e = self._embed_text(batch["text_ids"])             # (B, Tt, d)
        audio_e = self._embed_frames(batch["codes"])             # (B, Tf, d)
        B, Tt = batch["text_ids"].shape
        Tf = batch["codes"].shape[2]
        L = batch["text_mask"].sum(1)                            # text lens
        Fr = batch["frame_mask"].sum(1) - 1                      # real frame counts

        ain_e = batch.get("audio_in_embeds")                     # (B, Ta, d) | None
        if ain_e is not None:
            Ta = ain_e.shape[1]
            Ain = batch["audio_in_mask"].sum(1)                  # per-item audio-in lens
        else:
            Ta = 0
            Ain = torch.zeros(B, dtype=torch.long, device=text_e.device)

        W = Tt + Ta + Tf
        x = text_e.new_zeros(B, W, text_e.shape[-1])
        attn = torch.zeros(B, W, dtype=torch.long, device=x.device)
        ple = None
        if self.uses_ple:
            ple_text = self.backbone.get_per_layer_inputs(batch["text_ids"], None)
            ple = ple_text.new_zeros(B, W, *ple_text.shape[2:])
        out_start = []                                           # output region start per item
        for i in range(B):
            li, ai, fi = int(L[i]), int(Ain[i]), int(Fr[i])
            x[i, :li] = text_e[i, :li]
            if ai:
                x[i, li : li + ai] = ain_e[i, :ai]
            s = li + ai
            x[i, s : s + fi] = audio_e[i, :fi]
            attn[i, : s + fi] = 1
            if ple is not None:
                ple[i, :li] = ple_text[i, :li]                   # audio (in & out) get zero PLE
            out_start.append(s)
        kwargs = {"per_layer_inputs": ple} if ple is not None else {}
        h = self.backbone(
            inputs_embeds=x, attention_mask=attn, output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1]

        # c0: for item i, output frame t (0..Fr_i incl. EOS) is predicted from
        # position (out_start_i - 1 + t) -- identical to inference geometry,
        # with the audio-in context attended to via the prefix.
        pred_hs, c0_tgts, depth_hs, depth_codes = [], [], [], []
        c0_logits_padded = h.new_zeros(B, Tf, self.cfg.codebook_size + 1)
        for i in range(B):
            s, fi = out_start[i], int(Fr[i])
            pos = torch.arange(s - 1, s + fi, device=h.device)
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
                "loss": loss_c0 + self.cfg.depth_loss_weight * loss_depth,
                "c0_logits": c0_logits_padded}

    # -- joint text+audio (inner-monologue S2S) --------------------------------
    def _text_logits(self, h):
        """Backbone-hidden -> text-vocab logits via Gemma's tied LM head
        (lm_head.weight is embed_tokens.weight) + final_logit_softcapping."""
        W = self.backbone.get_input_embeddings().weight        # (V, d), tied
        logits = F.linear(h, W)
        cap = getattr(self.backbone.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward_joint(self, batch, eos_id: int, bos_id: int):
        """Joint S2S objective. Geometry per item:
            [BOS | query audio-in | response_text | <eos> | response_audio]
        The model GENERATES the response text (LM loss) AND the response audio
        (c0 + depth), so it learns to decide what to say from the query audio,
        then speak it. The <eos> token marks the text->audio handoff: its
        position predicts the first audio frame (out_start-1), and it is the
        final text target so the model learns to stop talking-in-text.

        Position 0 MUST be the BOS token (a real, calibrated embedding): Gemma
        relies on the first position as an attention sink with a massive,
        calibrated activation. Putting a raw audio-in embedding at position 0
        (no BOS) made that sink degenerate, so RMSNorm's rsqrt backward went
        singular -> NaN gradient -> the first opt.step poisoned every weight via
        the global grad-norm. The working Phase-B forward always started with
        text (hence BOS), which is why it never hit this.
        """
        text_e = self._embed_text(batch["text_ids"])           # (B, Tt, d) response text
        audio_e = self._embed_frames(batch["codes"])           # (B, Tf, d) response audio
        eos_e = self._embed_text(batch["text_ids"].new_full((1, 1), eos_id))[0, 0]  # (d,)
        bos_e = self._embed_text(batch["text_ids"].new_full((1, 1), bos_id))[0, 0]  # (d,) sink
        B, Tt = batch["text_ids"].shape
        Tf = batch["codes"].shape[2]
        L = batch["text_mask"].sum(1)                          # text lens (incl. BOS)
        Fr = batch["frame_mask"].sum(1) - 1                    # real frame counts
        ain_e = batch["audio_in_embeds"]                       # (B, Ta, d) query audio
        Ta = ain_e.shape[1]
        Ain = batch["audio_in_mask"].sum(1)

        W = 1 + Ta + Tt + 1 + Tf                               # +1 BOS sink, +1 <eos>
        x = text_e.new_zeros(B, W, text_e.shape[-1])
        attn = torch.zeros(B, W, dtype=torch.long, device=x.device)
        ple = None
        if self.uses_ple:
            ple_text = self.backbone.get_per_layer_inputs(batch["text_ids"], None)
            eos_ple = self.backbone.get_per_layer_inputs(
                batch["text_ids"].new_full((1, 1), eos_id), None)[0, 0]
            bos_ple = self.backbone.get_per_layer_inputs(
                batch["text_ids"].new_full((1, 1), bos_id), None)[0, 0]
            ple = ple_text.new_zeros(B, W, *ple_text.shape[2:])
        text_start, out_start = [], []
        for i in range(B):
            ai, li, fi = int(Ain[i]), int(L[i]), int(Fr[i])
            x[i, 0] = bos_e                                     # BOS attention sink at pos 0
            x[i, 1:1 + ai] = ain_e[i, :ai]
            ts = 1 + ai                                        # text begins after BOS+audio-in
            x[i, ts:ts + li] = text_e[i, :li]
            x[i, ts + li] = eos_e                              # <eos> handoff token
            s = ts + li + 1                                    # audio-out begins after <eos>
            x[i, s:s + fi] = audio_e[i, :fi]
            attn[i, :s + fi] = 1
            if ple is not None:
                ple[i, 0] = bos_ple                            # BOS is a text token -> real PLE
                ple[i, ts:ts + li] = ple_text[i, :li]
                ple[i, ts + li] = eos_ple                      # <eos> is a text token -> real PLE
            text_start.append(ts); out_start.append(s)
        kwargs = {"per_layer_inputs": ple} if ple is not None else {}
        h = self.backbone(inputs_embeds=x, attention_mask=attn,
                          output_hidden_states=True, **kwargs).hidden_states[-1]

        # debug: localize the first non-finite tensor (fires only when NaN/Inf present)
        for _nm, _t in (("audio_in_embeds", ain_e), ("text_e", text_e),
                        ("audio_e", audio_e), ("x_prefix", x), ("backbone_h", h)):
            if _t is not None and not torch.isfinite(_t).all():
                print(f"NONFINITE[{_nm}] "
                      f"(ain max {ain_e.abs().max().item() if ain_e is not None else 0:.1f}, "
                      f"x max {x.abs().max().item():.1f}, h max "
                      f"{h.abs().max().item() if torch.isfinite(h).all() else float('nan'):.1f})",
                      flush=True)
                break

        # text LM loss: positions [ts-1 .. ts+li-1] predict [text_0..text_{li-1}, <eos>]
        # skipped entirely when text_loss_weight==0 (isolation: NOT computed, so the
        # _text_logits/CE path can't run -- to test if that path is the corrupter)
        w_text = getattr(self.cfg, "text_loss_weight", 1.0)
        loss_text = h.new_zeros(())
        if w_text > 0:
            text_pred_h, text_tgts = [], []
            for i in range(B):
                ts, li = text_start[i], int(L[i])
                if li == 0:
                    continue
                pos = torch.arange(ts - 1, ts + li, device=h.device)   # li+1 positions
                text_pred_h.append(h[i, pos])
                text_tgts.append(torch.cat([batch["text_ids"][i, :li],
                                            batch["text_ids"].new_tensor([eos_id])]))
            if text_pred_h:
                loss_text = F.cross_entropy(
                    self._text_logits(torch.cat(text_pred_h)).float(), torch.cat(text_tgts))

        # c0 + depth on audio-out (out_start-1 = the <eos> position predicts frame 0)
        pred_hs, c0_tgts, depth_hs, depth_codes = [], [], [], []
        for i in range(B):
            s, fi = out_start[i], int(Fr[i])
            pos = torch.arange(s - 1, s + fi, device=h.device)
            pred_hs.append(h[i, pos])
            c0_tgts.append(torch.cat([batch["codes"][i, 0, :fi],
                                      batch["codes"].new_tensor([self.cfg.c0_eos_id])]))
            depth_hs.append(h[i, pos[:-1]])
            depth_codes.append(batch["codes"][i, :, :fi].T)
        loss_c0 = F.cross_entropy(self.c0_head(torch.cat(pred_hs)).float(), torch.cat(c0_tgts))
        all_h = torch.cat(depth_hs); all_codes = torch.cat(depth_codes)
        if len(all_h) == 0:
            loss_depth = loss_c0.new_zeros(())
        else:
            n_keep = max(1, int(len(all_h) * self.cfg.depth_loss_frame_frac))
            sel = torch.randperm(len(all_h))[:n_keep]
            loss_depth = self.depth.loss(all_h[sel], all_codes[sel])

        w_text = getattr(self.cfg, "text_loss_weight", 1.0)
        loss = w_text * loss_text + loss_c0 + self.cfg.depth_loss_weight * loss_depth
        return {"loss": loss, "loss_text": loss_text,
                "loss_c0": loss_c0, "loss_depth": loss_depth}

    # Gemma-4 Unified chat-template framing around the audio-in (extracted from the
    # processor): prefix = "<bos><|turn>user\n<|audio>";  suffix =
    # "<audio|><turn|>\n<|turn>model\n<|channel>thought<channel|>". Wrapping the
    # spliced audio in this native framing is what makes it convey lexical CONTENT
    # (not just prosody) -- see paper sec:delivery. Position 0 is BOS (attention sink).
    _UNI_PREFIX = (2, 105, 2364, 107, 256000)
    _UNI_SUFFIX = (258883, 106, 107, 105, 4368, 107, 100, 45518, 107, 101)

    def forward_unified(self, batch, eos_id: int):
        """Native-delivery joint S2S (segment-split). Per item:
          seg1 (FULL multimodal model): input_ids = [<bos>user\\n<audio> | <audio>*N
               | <audio>..model thought | reply_text | <eos>] delivered WITH
               input_features, so the backbone's own multimodal integration
               (masked_scatter + modality attention) carries lexical CONTENT -- the
               proven path (scripts/exp_native_delivery). Text LM loss on the reply
               positions teaches the model to DECIDE the reply from the audio.
          seg2 (backbone + KV from seg1): reply-audio frames -> c0/depth, so the
               speech head conditions on hidden states computed from the native
               audio context (Qwen Thinker->Talker), not a re-encoded text string.
        Loss geometry is identical to the old splice version; only delivery changed.
        Batch handled by looping (per-item audio length); use grad-accum for an
        effective batch. REQUIRES gradient checkpointing OFF (seg1 needs use_cache).
        Position 0 is <bos> (attention sink), never audio.
        """
        assert hasattr(self, "full_model"), \
            "forward_unified needs the full multimodal model (loader must set model.full_model)"
        dev = self.c0_head.weight.device
        dt = self.c0_head.weight.dtype
        atok = self.audio_token_id
        pre, suf = list(self._UNI_PREFIX), list(self._UNI_SUFFIX)
        P, S = len(pre), len(suf)
        feats_all = batch["input_features"].to(dev, dt)            # (B, Ta, 640) raw query feats
        fmask = batch["input_features_mask"].to(dev)               # (B, Ta)
        B = batch["text_ids"].shape[0]
        L = batch["text_mask"].sum(1)
        Fr = batch["frame_mask"].sum(1) - 1

        text_pred_h, text_tgts = [], []
        pred_hs, c0_tgts, depth_hs, depth_codes = [], [], [], []
        for i in range(B):
            li, fi, na = int(L[i]), int(Fr[i]), int(fmask[i].sum())
            reply = batch["text_ids"][i, :li].tolist()
            ids = pre + [atok] * na + suf + reply + [eos_id]
            ids_t = batch["text_ids"].new_tensor([ids])
            out1 = self.full_model(
                input_ids=ids_t,
                input_features=feats_all[i, :na].unsqueeze(0),
                input_features_mask=torch.ones(1, na, dtype=torch.bool, device=dev),
                output_hidden_states=True, use_cache=True)
            H1 = out1.hidden_states[-1][0]                         # (Lpre, d); eos = H1[-1]
            ts = P + na + S                                        # reply_text start index
            if li > 0:                                            # text LM loss: predict reply + eos
                text_pred_h.append(H1[ts - 1: ts + li])
                text_tgts.append(torch.cat([batch["text_ids"][i, :li],
                                            batch["text_ids"].new_tensor([eos_id])]))
            if fi > 0:                                            # seg2: frames via backbone + KV
                frame_e = self._embed_frames(batch["codes"][i, :, :fi].unsqueeze(0)).to(dt)
                H2 = self.backbone(inputs_embeds=frame_e, past_key_values=out1.past_key_values,
                                   output_hidden_states=True, use_cache=False).hidden_states[-1][0]
                ph = torch.cat([H1[-1:].to(dt), H2], dim=0)        # (fi+1, d): eos + frames 0..fi-1
            else:
                ph = H1[-1:].to(dt)
            pred_hs.append(ph)                                    # predict frames 0..fi-1 + c0_eos
            c0_tgts.append(torch.cat([batch["codes"][i, 0, :fi],
                                      batch["codes"].new_tensor([self.cfg.c0_eos_id])]))
            if fi > 0:
                depth_hs.append(ph[:fi])                          # states predicting frames 0..fi-1
                depth_codes.append(batch["codes"][i, :, :fi].T)

        w_text = getattr(self.cfg, "text_loss_weight", 1.0)
        loss_text = self.c0_head.weight.new_zeros(())
        if w_text > 0 and text_pred_h:
            loss_text = F.cross_entropy(
                self._text_logits(torch.cat(text_pred_h)).float(), torch.cat(text_tgts))
        loss_c0 = F.cross_entropy(self.c0_head(torch.cat(pred_hs)).float(), torch.cat(c0_tgts))
        if depth_hs:
            all_h = torch.cat(depth_hs); all_codes = torch.cat(depth_codes)
            n_keep = max(1, int(len(all_h) * self.cfg.depth_loss_frame_frac))
            sel = torch.randperm(len(all_h))[:n_keep]
            loss_depth = self.depth.loss(all_h[sel], all_codes[sel])
        else:
            loss_depth = loss_c0.new_zeros(())
        loss = w_text * loss_text + loss_c0 + self.cfg.depth_loss_weight * loss_depth
        return {"loss": loss, "loss_text": loss_text,
                "loss_c0": loss_c0, "loss_depth": loss_depth}

    def forward_grounded(self, batch, eos_id: int, bos_id: int):
        """Grounded inner monologue. Geometry per item:
            [BOS | query-audio | query_text | <eos> | response_text | <eos> | response_audio]
        Adds a COMPREHENSION step: the model first transcribes what it heard
        (query_text), then replies (response_text), then speaks. Forcing the
        audio content into the text channel before replying grounds the reply on
        the actual words (the audio embeddings alone don't deliver clean
        semantics to the frozen LM). Two text LM losses are logged separately:
          comprehension (predict query_text + <eos1>) and reply (predict
          response_text + <eos2>). Audio (c0/depth) on the final segment, with
          the <eos2> position predicting frame 0 -- identical to forward_joint.
        Mirrors forward_joint's BOS-sink fix (position 0 = BOS, never audio).
        """
        text_e = self._embed_text(batch["text_ids"])           # (B, Tt, d) response
        qtext_e = self._embed_text(batch["query_text_ids"])    # (B, Tq, d) transcript target
        audio_e = self._embed_frames(batch["codes"])           # (B, Tf, d) response audio
        eos_e = self._embed_text(batch["text_ids"].new_full((1, 1), eos_id))[0, 0]
        bos_e = self._embed_text(batch["text_ids"].new_full((1, 1), bos_id))[0, 0]
        B, Tt = batch["text_ids"].shape
        Tq = batch["query_text_ids"].shape[1]
        Tf = batch["codes"].shape[2]
        Lr = batch["text_mask"].sum(1)                         # reply lens
        Lq = batch["query_text_mask"].sum(1)                   # transcript lens
        Fr = batch["frame_mask"].sum(1) - 1
        ain_e = batch["audio_in_embeds"]
        Ta = ain_e.shape[1]
        Ain = batch["audio_in_mask"].sum(1)

        W = 1 + Ta + Tq + 1 + Tt + 1 + Tf                      # BOS + audio + qtext + eos1 + reply + eos2 + audio
        x = text_e.new_zeros(B, W, text_e.shape[-1])
        attn = torch.zeros(B, W, dtype=torch.long, device=x.device)
        ple = None
        if self.uses_ple:
            ple_text = self.backbone.get_per_layer_inputs(batch["text_ids"], None)
            ple_qtext = self.backbone.get_per_layer_inputs(batch["query_text_ids"], None)
            eos_ple = self.backbone.get_per_layer_inputs(
                batch["text_ids"].new_full((1, 1), eos_id), None)[0, 0]
            bos_ple = self.backbone.get_per_layer_inputs(
                batch["text_ids"].new_full((1, 1), bos_id), None)[0, 0]
            ple = ple_text.new_zeros(B, W, *ple_text.shape[2:])
        cmp_start, rep_start, out_start = [], [], []
        for i in range(B):
            ai, qi, ri, fi = int(Ain[i]), int(Lq[i]), int(Lr[i]), int(Fr[i])
            x[i, 0] = bos_e
            x[i, 1:1 + ai] = ain_e[i, :ai]
            cs = 1 + ai                                        # transcript begins after BOS+audio
            x[i, cs:cs + qi] = qtext_e[i, :qi]
            x[i, cs + qi] = eos_e                              # <eos1> transcript->reply
            rs = cs + qi + 1                                   # reply begins after <eos1>
            x[i, rs:rs + ri] = text_e[i, :ri]
            x[i, rs + ri] = eos_e                              # <eos2> reply->audio handoff
            s = rs + ri + 1                                    # audio-out begins after <eos2>
            x[i, s:s + fi] = audio_e[i, :fi]
            attn[i, :s + fi] = 1
            if ple is not None:
                ple[i, 0] = bos_ple
                ple[i, cs:cs + qi] = ple_qtext[i, :qi]
                ple[i, cs + qi] = eos_ple
                ple[i, rs:rs + ri] = ple_text[i, :ri]
                ple[i, rs + ri] = eos_ple
            cmp_start.append(cs); rep_start.append(rs); out_start.append(s)
        kwargs = {"per_layer_inputs": ple} if ple is not None else {}
        h = self.backbone(inputs_embeds=x, attention_mask=attn,
                          output_hidden_states=True, **kwargs).hidden_states[-1]

        def _lm_loss(starts, ids, lens, eid):
            ph, tg = [], []
            for i in range(B):
                st, li = starts[i], int(lens[i])
                if li == 0:
                    continue
                pos = torch.arange(st - 1, st + li, device=h.device)   # li+1 positions
                ph.append(h[i, pos])
                tg.append(torch.cat([ids[i, :li], ids.new_tensor([eid])]))
            if not ph:
                return h.new_zeros(())
            return F.cross_entropy(self._text_logits(torch.cat(ph)).float(), torch.cat(tg))

        w_cmp = getattr(self.cfg, "comprehension_loss_weight", 1.0)
        w_rep = getattr(self.cfg, "text_loss_weight", 1.0)
        loss_cmp = _lm_loss(cmp_start, batch["query_text_ids"], Lq, eos_id) if w_cmp > 0 else h.new_zeros(())
        loss_rep = _lm_loss(rep_start, batch["text_ids"], Lr, eos_id) if w_rep > 0 else h.new_zeros(())

        # c0 + depth on audio-out (out_start-1 = <eos2> predicts frame 0)
        pred_hs, c0_tgts, depth_hs, depth_codes = [], [], [], []
        for i in range(B):
            s, fi = out_start[i], int(Fr[i])
            pos = torch.arange(s - 1, s + fi, device=h.device)
            pred_hs.append(h[i, pos])
            c0_tgts.append(torch.cat([batch["codes"][i, 0, :fi],
                                      batch["codes"].new_tensor([self.cfg.c0_eos_id])]))
            depth_hs.append(h[i, pos[:-1]])
            depth_codes.append(batch["codes"][i, :, :fi].T)
        loss_c0 = F.cross_entropy(self.c0_head(torch.cat(pred_hs)).float(), torch.cat(c0_tgts))
        all_h = torch.cat(depth_hs); all_codes = torch.cat(depth_codes)
        if len(all_h) == 0:
            loss_depth = loss_c0.new_zeros(())
        else:
            n_keep = max(1, int(len(all_h) * self.cfg.depth_loss_frame_frac))
            sel = torch.randperm(len(all_h))[:n_keep]
            loss_depth = self.depth.loss(all_h[sel], all_codes[sel])

        loss = (w_cmp * loss_cmp + w_rep * loss_rep + loss_c0
                + self.cfg.depth_loss_weight * loss_depth)
        return {"loss": loss, "loss_compr": loss_cmp, "loss_text": loss_rep,
                "loss_c0": loss_c0, "loss_depth": loss_depth}

    # -- inference ---------------------------------------------------------------
    @torch.no_grad()
    def generate_codes(self, text_ids, max_frames: int = 1000,
                       temperature: float = 0.9, top_k: int = 0,
                       audio_in_embeds=None, min_frames: int = 0):
        """text_ids: (1, Tt) -> (32, T) codes.

        Optional audio_in_embeds (1, Ta, d) is the encoder-free audio-IN segment
        (from loader.audio_encode); when given, the prefix becomes
        [text | audio-in] -- identical geometry to the training forward -- so the
        spoken output conditions on the input audio (audio-in -> audio-out).

        Uses a KV cache when the backbone has no per-layer inputs (the 12B): the
        prefix is encoded once, then each frame feeds only the new embedding, so
        per-frame cost is O(1) in sequence length instead of O(n). PLE backbones
        (E2B) fall back to the no-cache full-recompute path."""
        x = self._embed_text(text_ids)                            # (1, Tt, d)
        n_pre = 0
        if audio_in_embeds is not None:
            ain = audio_in_embeds.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, ain], dim=1)                        # [text | audio-in]
            n_pre = ain.shape[1]
        use_cache = not self.uses_ple
        frames, past, cur = [], None, x
        for _ in range(max_frames):
            if use_cache:
                out = self.backbone(inputs_embeds=cur, past_key_values=past,
                                    use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                h = out.hidden_states[-1][:, -1]                  # (1, d)
            else:
                h = self.backbone(
                    inputs_embeds=cur, output_hidden_states=True,
                    **self._backbone_kwargs(text_ids, n_pre + len(frames)),
                ).hidden_states[-1][:, -1]
            logits = self.c0_head(h).float() / temperature
            if len(frames) < min_frames:
                logits[..., self.cfg.c0_eos_id] = float("-inf")   # ban early stop
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)                    # (1, 32)
            frames.append(frame[0])
            nxt = self._embed_frames(frame.unsqueeze(-1))         # (1, 1, d)
            cur = nxt if use_cache else torch.cat([cur, nxt], dim=1)
        if not frames:
            return torch.zeros(32, 1, dtype=torch.long)
        return torch.stack(frames, dim=1)                         # (32, T)

    @torch.no_grad()
    def generate_joint(self, audio_in_embeds, tok, eos_id, max_text: int = 96,
                       max_frames: int = 400, temperature: float = 0.9, top_k: int = 50,
                       text_temperature: float = 0.7, min_frames: int = 0):
        """Inference for --joint: [query audio-in] -> GENERATE response text (until
        <eos>) -> GENERATE response audio frames. Returns (text_str, codes (32,T)).
        12B (no PLE) only -- uses a KV cache; mirrors forward_joint's geometry."""
        assert not self.uses_ple, "generate_joint is KV-cache only (12B); PLE backbones unsupported"
        x = audio_in_embeds.to(dtype=self.c0_head.weight.dtype, device=self.c0_head.weight.device)
        bos = self._embed_text(torch.tensor([[tok.bos_token_id]], device=x.device))  # (1,1,d)
        x = torch.cat([bos.to(x.dtype), x], dim=1)               # [BOS | audio-in] -> matches training
        out = self.backbone(inputs_embeds=x, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1]                          # predicts text token 0

        # phase 1: response text
        text_ids = []
        for _ in range(max_text):
            logits = self._text_logits(h).float() / max(text_temperature, 1e-5)
            nid = int(torch.multinomial(F.softmax(logits, -1), 1)) if text_temperature > 0 \
                else int(logits.argmax(-1))
            emb = self._embed_text(torch.tensor([[nid]], device=h.device))
            out = self.backbone(inputs_embeds=emb, past_key_values=past,
                                use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[-1][:, -1]
            if nid == eos_id:
                break                                             # h is now the <eos> hidden -> frame 0
            text_ids.append(nid)
        text_str = tok.decode(text_ids).strip()

        # phase 2: response audio (h = post-<eos> hidden predicts frame 0)
        frames = []
        for _ in range(max_frames):
            logits = self.c0_head(h).float() / temperature
            if len(frames) < min_frames:
                logits[..., self.cfg.c0_eos_id] = float("-inf")
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)                    # (1, 32)
            frames.append(frame[0])
            nxt = self._embed_frames(frame.unsqueeze(-1))
            out = self.backbone(inputs_embeds=nxt, past_key_values=past,
                                use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[-1][:, -1]
        codes = torch.stack(frames, dim=1) if frames else torch.zeros(32, 1, dtype=torch.long)
        return text_str, codes

    @torch.no_grad()
    def generate_unified(self, input_features, tok, eos_id, input_features_mask=None,
                         max_text: int = 96, max_frames: int = 400,
                         temperature: float = 0.9, top_k: int = 50,
                         text_temperature: float = 0.7, min_frames: int = 0,
                         force_text_ids=None):
        """Inference for --unified (NATIVE delivery): deliver the query audio through
        the FULL multimodal model (input_ids with <audio> placeholders +
        input_features -> the model's own masked_scatter, proven to carry content)
        -> GENERATE response text (until <eos>) -> GENERATE response audio via the
        head. Mirrors forward_unified's seg1->seg2; the text/audio loops below are
        unchanged (KV-cache on self.backbone, the same underlying language_model)."""
        assert hasattr(self, "full_model"), "generate_unified needs model.full_model (native delivery)"
        dev = self.c0_head.weight.device; dt = self.c0_head.weight.dtype
        feats = input_features.to(dev, dt)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        na = feats.shape[1] if input_features_mask is None else int(input_features_mask.sum())
        feats = feats[:, :na]
        ids = list(self._UNI_PREFIX) + [self.audio_token_id] * na + list(self._UNI_SUFFIX)
        out = self.full_model(
            input_ids=torch.tensor([ids], device=dev), input_features=feats,
            input_features_mask=torch.ones(1, na, dtype=torch.bool, device=dev),
            use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1]                          # predicts response text token 0

        if force_text_ids is None:
            text_ids = []
            for _ in range(max_text):
                logits = self._text_logits(h).float() / max(text_temperature, 1e-5)
                nid = int(torch.multinomial(F.softmax(logits, -1), 1)) if text_temperature > 0 \
                    else int(logits.argmax(-1))
                emb = self._embed_text(torch.tensor([[nid]], device=h.device))
                out = self.backbone(inputs_embeds=emb, past_key_values=past,
                                    use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                h = out.hidden_states[-1][:, -1]
                if nid == eos_id:
                    break
                text_ids.append(nid)
            text_str = tok.decode(text_ids).strip()
        else:
            # Hybrid voicing: teacher-force a reply chosen elsewhere (e.g. by the
            # pristine LoRA-off backbone), advancing the KV/h exactly as the sampling
            # path would (reply tokens then a final <eos>) so the head's first frame
            # sees the same post-eos hidden state it normally does. The audio loop
            # below then voices it at whatever LoRA strength is currently set.
            text_ids = list(force_text_ids)
            for nid in text_ids + [eos_id]:
                emb = self._embed_text(torch.tensor([[nid]], device=h.device))
                out = self.backbone(inputs_embeds=emb, past_key_values=past,
                                    use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                h = out.hidden_states[-1][:, -1]
            text_str = tok.decode(text_ids).strip()

        frames = []
        for _ in range(max_frames):
            logits = self.c0_head(h).float() / temperature
            if len(frames) < min_frames:
                logits[..., self.cfg.c0_eos_id] = float("-inf")
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)
            frames.append(frame[0])
            nxt = self._embed_frames(frame.unsqueeze(-1))
            out = self.backbone(inputs_embeds=nxt, past_key_values=past,
                                use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[-1][:, -1]
        codes = torch.stack(frames, dim=1) if frames else torch.zeros(32, 1, dtype=torch.long)
        return text_str, codes

    @torch.no_grad()
    def generate_grounded(self, audio_in_embeds, tok, eos_id, max_text: int = 96,
                          max_frames: int = 400, temperature: float = 0.9, top_k: int = 50,
                          text_temperature: float = 0.7, min_frames: int = 0):
        """Inference for --grounded: [BOS | audio-in] -> TRANSCRIBE (until <eos>)
        -> REPLY (until <eos>) -> AUDIO. Returns (heard_text, reply_text, codes).
        12B (no PLE) only -- KV cache; mirrors forward_grounded's geometry."""
        assert not self.uses_ple, "generate_grounded is KV-cache only (12B)"
        x = audio_in_embeds.to(dtype=self.c0_head.weight.dtype, device=self.c0_head.weight.device)
        bos = self._embed_text(torch.tensor([[tok.bos_token_id]], device=x.device))
        x = torch.cat([bos.to(x.dtype), x], dim=1)
        out = self.backbone(inputs_embeds=x, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1]

        def gen_text():
            nonlocal past, h
            ids = []
            for _ in range(max_text):
                logits = self._text_logits(h).float() / max(text_temperature, 1e-5)
                nid = (int(torch.multinomial(F.softmax(logits, -1), 1))
                       if text_temperature > 0 else int(logits.argmax(-1)))
                emb = self._embed_text(torch.tensor([[nid]], device=h.device))
                out = self.backbone(inputs_embeds=emb, past_key_values=past,
                                    use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                h = out.hidden_states[-1][:, -1]
                if nid == eos_id:
                    break
                ids.append(nid)
            return tok.decode(ids).strip()

        heard_text = gen_text()   # phase 1: transcribe what was heard
        reply_text = gen_text()   # phase 2: reply, grounded on the transcript

        frames = []               # phase 3: speak the reply
        for _ in range(max_frames):
            logits = self.c0_head(h).float() / temperature
            if len(frames) < min_frames:
                logits[..., self.cfg.c0_eos_id] = float("-inf")
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            c0 = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            if c0.item() == self.cfg.c0_eos_id:
                break
            frame = self.depth.generate(h, c0)
            frames.append(frame[0])
            nxt = self._embed_frames(frame.unsqueeze(-1))
            out = self.backbone(inputs_embeds=nxt, past_key_values=past,
                                use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[-1][:, -1]
        codes = torch.stack(frames, dim=1) if frames else torch.zeros(32, 1, dtype=torch.long)
        return heard_text, reply_text, codes

    @torch.no_grad()
    def generate_wav(self, text_ids, codec, max_frames: int = 1000,
                     temperature: float = 0.9, top_k: int = 0,
                     min_rms: float = 0.01, retries: int = 3,
                     voiced_rms: float = 0.02, max_trail_s: float = 1.5,
                     pad_s: float = 0.2, sr: int = 24000, audio_in_embeds=None,
                     min_frames: int = 0):
        """generate_codes + decode, with silence/run-on handling + trailing trim.

        Catches two failure modes and trims trailing silence on success:
          (a) 'silent attractor' -- the whole clip is ~silent (valid non-EOS c0
              tokens that decode to silence and run to max_frames);
          (b) 'run-on' -- a few seconds of speech then a long silent tail because
              the model never emitted EOS (the 12B's failure mode; whole-clip RMS
              passes, so the old min_rms check missed it).
        On a clip that ends cleanly we trim the trailing silence to where speech
        stops (keeping a short pad). We re-sample (temperature nudged up) when the
        clip is fully silent or runs on. Returns (wav, codes, attempts).
        """
        win = max(1, int(sr * 0.08))                  # 80 ms = one Mimi frame
        pad = int(sr * pad_s)
        best = None
        for attempt in range(retries):
            t = temperature * (1.0 + 0.15 * attempt)
            codes = self.generate_codes(text_ids, max_frames=max_frames,
                                        temperature=t, top_k=top_k,
                                        audio_in_embeds=audio_in_embeds,
                                        min_frames=min_frames)
            wav = codec.decode(codes)
            w = wav.detach().float().reshape(-1)
            rms = w.pow(2).mean().sqrt().item()
            nwin = max(1, (w.numel() + win - 1) // win)
            wp = F.pad(w, (0, nwin * win - w.numel()))
            wrms = wp.reshape(nwin, win).pow(2).mean(-1).sqrt()
            voiced = (wrms >= voiced_rms).nonzero()
            if rms < min_rms or voiced.numel() == 0:
                best = (w[:win].reshape(1, -1), codes, attempt + 1)
                continue                              # fully silent -> retry
            end = min(w.numel(), (int(voiced[-1].item()) + 1) * win)
            trail_s = (w.numel() - end) / sr
            trimmed = w[:min(w.numel(), end + pad)].reshape(1, -1)
            best = (trimmed, codes, attempt + 1)      # trimmed fallback
            if trail_s <= max_trail_s:
                return best                           # clean ending -> done
            # run-on (long silent tail): retry for a cleaner take
        return best  # give up; return best (trimmed) attempt rather than hang
