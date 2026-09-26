from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class ComputerUseConfig:
    byte_vocab: int = 256
    task_len: int = 128
    action_len: int = 96
    image_size: int = 224
    patch: int = 16
    embd: int = 256
    text_layers: int = 2
    vision_layers: int = 2
    n_head: int = 8
    num_ops: int = 3

class ComputerUseModel(nn.Module):
    """Screenshot + natural-language task -> operation type + action text."""

    def __init__(self, cfg: ComputerUseConfig):
        super().__init__()
        self.cfg = cfg

        self.task_emb = nn.Embedding(cfg.byte_vocab, cfg.embd)
        self.task_pos = nn.Embedding(cfg.task_len, cfg.embd)
        text_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embd,
            nhead=cfg.n_head,
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
            batch_first=True,
            norm_first=True,
        )
        self.vision_encoder = nn.TransformerEncoder(
            vision_layer, cfg.vision_layers
        )

        self.fuse = nn.Sequential(
            nn.Linear(cfg.embd * 2, cfg.embd),
            nn.GELU(),
            nn.LayerNorm(cfg.embd),
        )
        self.op_head = nn.Linear(cfg.embd, cfg.num_ops)

        self.action_emb = nn.Embedding(cfg.byte_vocab, cfg.embd)
        self.action_gru = nn.GRU(
            cfg.embd, cfg.embd, num_layers=2, batch_first=True
        )
        self.action_head = nn.Linear(cfg.embd, cfg.byte_vocab)

    def encode(self, images: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        t = task.size(1)
        pos = torch.arange(t, device=task.device)
        tx = self.task_emb(task) + self.task_pos(pos)[None]
        tx = self.text_encoder(tx).mean(dim=1)

        vx = self.patchify(images).flatten(2).transpose(1, 2)
        vx = self.vision_encoder(
            vx + self.vision_pos[:, : vx.size(1)]
        ).mean(dim=1)

        return self.fuse(torch.cat([vx, tx], dim=-1))

    def forward(
        self,
        images: torch.Tensor,
        task: torch.Tensor,
        action_in: torch.Tensor,
        op_target: torch.Tensor | None = None,
        action_target: torch.Tensor | None = None,
    ):
        ctx = self.encode(images, task)
        op_logits = self.op_head(ctx)

        dec = self.action_emb(action_in) + ctx[:, None, :]
        dec, _ = self.action_gru(dec)
        action_logits = self.action_head(dec)

        loss = None
        if op_target is not None and action_target is not None:
            op_loss = F.cross_entropy(op_logits, op_target)
            action_loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.size(-1)),
                action_target.reshape(-1),
            )
            loss = op_loss + action_loss

        return op_logits, action_logits, loss
