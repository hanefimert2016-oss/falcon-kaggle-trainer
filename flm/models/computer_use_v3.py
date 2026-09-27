from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ComputerUseV3Config:
    text_vocab: int = 16384
    task_pad_token: int = 0
    task_len: int = 160
    image_size: int = 224
    patch: int = 16
    embd: int = 768
    text_layers: int = 4
    vision_layers: int = 8
    n_head: int = 12
    num_ops: int = 12
    num_domains: int = 256
    payload_vocab: int = 259
    payload_bos: int = 256
    payload_eos: int = 257
    payload_pad: int = 258
    payload_len: int = 128
    dropout: float = 0.0


class ComputerUseV3(nn.Module):
    """Closed-loop GUI policy: screenshot + task -> one executable OS action.

    Coordinates are learned as a patch-pointer distribution instead of direct
    regression, avoiding the v0.5/v0.6 tendency to predict an average point.
    Payload bytes carry typed text/hotkeys while the op head selects semantics.
    """

    def __init__(self, cfg: ComputerUseV3Config):
        super().__init__()
        self.cfg = cfg
        if cfg.embd % cfg.n_head:
            raise ValueError("embd must be divisible by n_head")
        if cfg.image_size % cfg.patch:
            raise ValueError("image_size must be divisible by patch")

        self.task_emb = nn.Embedding(cfg.text_vocab, cfg.embd)
        self.task_pos = nn.Embedding(cfg.task_len, cfg.embd)
        text_layer = nn.TransformerEncoderLayer(
            cfg.embd, cfg.n_head, 4 * cfg.embd, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, cfg.text_layers)

        self.patchify = nn.Conv2d(3, cfg.embd, cfg.patch, cfg.patch)
        side = cfg.image_size // cfg.patch
        n_patches = side * side
        self.vision_pos = nn.Parameter(torch.zeros(1, n_patches, cfg.embd))
        vision_layer = nn.TransformerEncoderLayer(
            cfg.embd, cfg.n_head, 4 * cfg.embd, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.vision_encoder = nn.TransformerEncoder(vision_layer, cfg.vision_layers)

        self.task_to_vision = nn.MultiheadAttention(
            cfg.embd, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.fuse = nn.Sequential(
            nn.Linear(cfg.embd * 3, cfg.embd),
            nn.GELU(),
            nn.LayerNorm(cfg.embd),
        )
        self.op_head = nn.Linear(cfg.embd, cfg.num_ops)
        self.domain_head = nn.Linear(cfg.embd, cfg.num_domains)

        self.pointer_q = nn.Linear(cfg.embd, cfg.embd, bias=False)
        self.pointer2_q = nn.Linear(cfg.embd, cfg.embd, bias=False)

        self.payload_emb = nn.Embedding(cfg.payload_vocab, cfg.embd)
        self.payload_gru = nn.GRU(
            cfg.embd, cfg.embd, num_layers=2, batch_first=True, dropout=cfg.dropout
        )
        self.payload_init = nn.Linear(cfg.embd, 2 * cfg.embd)
        self.payload_head = nn.Linear(cfg.embd, cfg.payload_vocab)

        centers = []
        for row in range(side):
            for col in range(side):
                centers.append(((col + 0.5) / side, (row + 0.5) / side))
        self.register_buffer("patch_centers", torch.tensor(centers, dtype=torch.float32), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def encode(self, images: torch.Tensor, task: torch.Tensor):
        t = task.size(1)
        if t > self.cfg.task_len:
            raise ValueError("task sequence too long")
        pos = torch.arange(t, device=task.device)
        pad = task.eq(self.cfg.task_pad_token)
        tx = self.task_emb(task) + self.task_pos(pos)[None]
        tx = self.text_encoder(tx, src_key_padding_mask=pad)
        valid = (~pad).to(tx.dtype).unsqueeze(-1)
        task_pool = (tx * valid).sum(1) / valid.sum(1).clamp_min(1.0)

        vx = self.patchify(images).flatten(2).transpose(1, 2)
        vx = self.vision_encoder(vx + self.vision_pos[:, : vx.size(1)])
        vision_pool = vx.mean(1)
        grounded, attn = self.task_to_vision(task_pool[:, None], vx, vx, need_weights=True)
        ctx = self.fuse(torch.cat([task_pool, vision_pool, grounded[:, 0]], dim=-1))
        return ctx, vx, attn

    def _pointer(self, ctx, vx, head):
        q = head(ctx)
        return torch.einsum("bd,bpd->bp", q, vx) / math.sqrt(self.cfg.embd)

    def _coords_from_logits(self, logits):
        probs = logits.float().softmax(-1)
        centers = self.patch_centers.to(probs.device, probs.dtype)
        return probs @ centers

    def _payload_hidden(self, ctx):
        b = ctx.size(0)
        return self.payload_init(ctx).view(b, 2, self.cfg.embd).transpose(0, 1).contiguous()

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        images,
        task,
        payload_in,
        *,
        op_target=None, op_valid=None,
        pointer_target=None, pointer_valid=None,
        pointer2_target=None, pointer2_valid=None,
        domain_target=None, domain_valid=None,
        payload_target=None, payload_valid=None,
    ):
        ctx, vx, attention = self.encode(images, task)
        op_logits = self.op_head(ctx)
        domain_logits = self.domain_head(ctx)
        pointer_logits = self._pointer(ctx, vx, self.pointer_q)
        pointer2_logits = self._pointer(ctx, vx, self.pointer2_q)
        coord = self._coords_from_logits(pointer_logits)
        coord2 = self._coords_from_logits(pointer2_logits)

        emb = self.payload_emb(payload_in)
        dec, _ = self.payload_gru(emb, self._payload_hidden(ctx))
        payload_logits = self.payload_head(dec)

        zero = ctx.sum() * 0.0
        losses = {}
        if op_target is not None and op_valid is not None:
            losses["op"] = self._masked_mean(F.cross_entropy(op_logits, op_target, reduction="none"), op_valid)
        else:
            losses["op"] = zero
        if pointer_target is not None and pointer_valid is not None:
            losses["pointer"] = self._masked_mean(F.cross_entropy(pointer_logits, pointer_target, reduction="none"), pointer_valid)
        else:
            losses["pointer"] = zero
        if pointer2_target is not None and pointer2_valid is not None:
            losses["pointer2"] = self._masked_mean(F.cross_entropy(pointer2_logits, pointer2_target, reduction="none"), pointer2_valid)
        else:
            losses["pointer2"] = zero
        if domain_target is not None and domain_valid is not None:
            losses["domain"] = self._masked_mean(F.cross_entropy(domain_logits, domain_target, reduction="none"), domain_valid)
        else:
            losses["domain"] = zero
        if payload_target is not None and payload_valid is not None:
            tok = F.cross_entropy(
                payload_logits.transpose(1, 2), payload_target,
                ignore_index=-100, reduction="none",
            )
            tm = payload_target.ne(-100).to(tok.dtype)
            per = (tok * tm).sum(-1) / tm.sum(-1).clamp_min(1.0)
            losses["payload"] = self._masked_mean(per, payload_valid)
        else:
            losses["payload"] = zero

        loss = (
            losses["op"]
            + 2.0 * losses["pointer"]
            + 1.0 * losses["pointer2"]
            + 0.20 * losses["domain"]
            + losses["payload"]
        )
        outputs = {
            "op_logits": op_logits,
            "domain_logits": domain_logits,
            "pointer_logits": pointer_logits,
            "pointer2_logits": pointer2_logits,
            "coord": coord,
            "coord2": coord2,
            "payload_logits": payload_logits,
            "attention": attention,
        }
        return outputs, loss, losses

    @torch.no_grad()
    def generate_payload(self, ctx: torch.Tensor, max_new: int | None = None) -> list[bytes]:
        self.eval()
        limit = min(max_new or self.cfg.payload_len, self.cfg.payload_len)
        batch = ctx.size(0)
        hidden = self._payload_hidden(ctx)
        token = torch.full((batch, 1), self.cfg.payload_bos, dtype=torch.long, device=ctx.device)
        done = torch.zeros(batch, dtype=torch.bool, device=ctx.device)
        raw = [bytearray() for _ in range(batch)]
        for _ in range(limit):
            emb = self.payload_emb(token)
            dec, hidden = self.payload_gru(emb, hidden)
            logits = self.payload_head(dec[:, -1])
            nxt = logits.argmax(-1)
            for i, value in enumerate(nxt.tolist()):
                if done[i]:
                    continue
                if value == self.cfg.payload_eos:
                    done[i] = True
                elif 0 <= value <= 255:
                    raw[i].append(value)
            token = nxt[:, None]
            if bool(done.all()):
                break
        return [bytes(x) for x in raw]

    @torch.no_grad()
    def predict(self, images: torch.Tensor, task: torch.Tensor):
        self.eval()
        ctx, vx, _ = self.encode(images, task)
        op = self.op_head(ctx).argmax(-1)
        domain = self.domain_head(ctx).argmax(-1)
        p1 = self._pointer(ctx, vx, self.pointer_q)
        p2 = self._pointer(ctx, vx, self.pointer2_q)
        payload = [x.decode("utf-8", "replace") for x in self.generate_payload(ctx)]
        return {
            "op": op,
            "domain": domain,
            "coord": self._coords_from_logits(p1),
            "coord2": self._coords_from_logits(p2),
            "payload": payload,
        }
