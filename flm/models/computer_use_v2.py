from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ComputerUseV2Config:
    byte_vocab: int = 259
    bos_token: int = 256
    eos_token: int = 257
    pad_token: int = 258
    task_len: int = 192
    action_len: int = 128
    image_size: int = 224
    patch: int = 16
    embd: int = 768
    text_layers: int = 4
    vision_layers: int = 8
    n_head: int = 12
    num_ops: int = 8
    num_domains: int = 128
    dropout: float = 0.0


class ComputerUseV2(nn.Module):
    """From-scratch multimodal GUI/game agent.

    Screenshot + task -> operation, coordinate, bbox, domain and action text.
    Different real datasets can supervise different heads through validity masks.
    """

    def __init__(self, cfg: ComputerUseV2Config):
        super().__init__()
        self.cfg = cfg
        if cfg.embd % cfg.n_head:
            raise ValueError("embd must be divisible by n_head")
        if cfg.image_size % cfg.patch:
            raise ValueError("image_size must be divisible by patch")

        self.task_emb = nn.Embedding(cfg.byte_vocab, cfg.embd)
        self.task_pos = nn.Embedding(cfg.task_len, cfg.embd)
        text_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.embd,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, cfg.text_layers)

        self.patchify = nn.Conv2d(
            3, cfg.embd, kernel_size=cfg.patch, stride=cfg.patch
        )
        n_patches = (cfg.image_size // cfg.patch) ** 2
        self.vision_pos = nn.Parameter(torch.zeros(1, n_patches, cfg.embd))
        vision_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.embd,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vision_encoder = nn.TransformerEncoder(
            vision_layer, cfg.vision_layers
        )

        self.task_to_vision = nn.MultiheadAttention(
            cfg.embd,
            cfg.n_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(cfg.embd * 3, cfg.embd),
            nn.GELU(),
            nn.LayerNorm(cfg.embd),
        )

        self.op_head = nn.Linear(cfg.embd, cfg.num_ops)
        self.coord_head = nn.Sequential(nn.Linear(cfg.embd, 2), nn.Sigmoid())
        self.bbox_head = nn.Sequential(nn.Linear(cfg.embd, 4), nn.Sigmoid())
        self.domain_head = nn.Linear(cfg.embd, cfg.num_domains)

        self.action_emb = nn.Embedding(cfg.byte_vocab, cfg.embd)
        self.action_gru = nn.GRU(
            cfg.embd,
            cfg.embd,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.action_head = nn.Linear(cfg.embd, cfg.byte_vocab)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            if getattr(module, "weight", None) is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self, images: torch.Tensor, task: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = task.size(1)
        pos = torch.arange(t, device=task.device)
        task_pad = task.eq(self.cfg.pad_token)
        tx = self.task_emb(task) + self.task_pos(pos)[None, :, :]
        tx = self.text_encoder(tx, src_key_padding_mask=task_pad)
        valid = (~task_pad).to(tx.dtype).unsqueeze(-1)
        task_pool = (tx * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        vx = self.patchify(images).flatten(2).transpose(1, 2)
        vx = self.vision_encoder(
            vx + self.vision_pos[:, : vx.size(1), :]
        )
        vision_pool = vx.mean(dim=1)

        query = task_pool[:, None, :]
        grounded, attn = self.task_to_vision(query, vx, vx, need_weights=True)
        grounded = grounded[:, 0, :]
        ctx = self.fuse(torch.cat([task_pool, vision_pool, grounded], dim=-1))
        return ctx, attn

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        images: torch.Tensor,
        task: torch.Tensor,
        action_in: torch.Tensor,
        *,
        op_target: torch.Tensor | None = None,
        op_valid: torch.Tensor | None = None,
        coord_target: torch.Tensor | None = None,
        coord_valid: torch.Tensor | None = None,
        bbox_target: torch.Tensor | None = None,
        bbox_valid: torch.Tensor | None = None,
        domain_target: torch.Tensor | None = None,
        domain_valid: torch.Tensor | None = None,
        action_target: torch.Tensor | None = None,
        action_valid: torch.Tensor | None = None,
    ):
        ctx, attention = self.encode(images, task)
        op_logits = self.op_head(ctx)
        coord = self.coord_head(ctx)
        bbox = self.bbox_head(ctx)
        domain_logits = self.domain_head(ctx)

        dec = self.action_emb(action_in) + ctx[:, None, :]
        dec, _ = self.action_gru(dec)
        action_logits = self.action_head(dec)

        # Always return the same loss keys on every device.
        # DataParallel gathers dictionaries from each GPU and requires identical
        # key sets. Different datasets supervise different heads, so a shard can
        # legitimately have zero valid samples for a head. In that case return
        # a differentiable zero instead of omitting the key.
        zero = ctx.sum() * 0.0
        losses: dict[str, torch.Tensor] = {}

        if op_target is not None and op_valid is not None:
            per = F.cross_entropy(op_logits, op_target, reduction="none")
            losses["op"] = self._masked_mean(per, op_valid)
        else:
            losses["op"] = zero

        if coord_target is not None and coord_valid is not None:
            per = F.smooth_l1_loss(coord, coord_target, reduction="none").mean(-1)
            losses["coord"] = self._masked_mean(per, coord_valid)
        else:
            losses["coord"] = zero

        if bbox_target is not None and bbox_valid is not None:
            per = F.smooth_l1_loss(bbox, bbox_target, reduction="none").mean(-1)
            losses["bbox"] = self._masked_mean(per, bbox_valid)
        else:
            losses["bbox"] = zero

        if domain_target is not None and domain_valid is not None:
            per = F.cross_entropy(domain_logits, domain_target, reduction="none")
            losses["domain"] = self._masked_mean(per, domain_valid)
        else:
            losses["domain"] = zero

        if action_target is not None and action_valid is not None:
            token_loss = F.cross_entropy(
                action_logits.transpose(1, 2),
                action_target,
                ignore_index=-100,
                reduction="none",
            )
            token_mask = action_target.ne(-100).to(token_loss.dtype)
            per_sample = (token_loss * token_mask).sum(-1) / token_mask.sum(-1).clamp_min(1.0)
            losses["action"] = self._masked_mean(per_sample, action_valid)
        else:
            losses["action"] = zero

        weights = {
            "op": 1.0,
            "coord": 2.0,
            "bbox": 2.0,
            "domain": 0.25,
            "action": 1.0,
        }
        loss = sum(weights[k] * losses[k] for k in ("op", "coord", "bbox", "domain", "action"))

        outputs = {
            "op_logits": op_logits,
            "coord": coord,
            "bbox": bbox,
            "domain_logits": domain_logits,
            "action_logits": action_logits,
            "attention": attention,
        }
        return outputs, loss, losses
