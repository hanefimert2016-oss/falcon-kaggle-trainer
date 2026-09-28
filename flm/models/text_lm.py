from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F

@dataclass
class TextConfig:
    vocab_size: int = 256
    seq_len: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    position_encoding: str = "learned"

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.position_encoding = cfg.position_encoding
        if self.position_encoding == "rope":
            if self.head_dim % 2:
                raise ValueError("RoPE requires an even attention head dimension")
            inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("rope_inv_freq", inv, persistent=False)
        elif self.position_encoding != "learned":
            raise ValueError(f"unknown position encoding: {self.position_encoding}")

    def _rope(self, q, k):
        t = q.size(-2)
        pos = torch.arange(t, device=q.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(pos, self.rope_inv_freq)
        cos = freqs.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype)[None, None, :, :]

        def rotate(x):
            xe = x[..., 0::2]
            xo = x[..., 1::2]
            return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)

    def forward(self, x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).split(c,dim=-1)
        q=q.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        if self.position_encoding == "rope":
            q, k = self._rope(q, k)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.n_embd)
        self.attn=CausalSelfAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.n_embd)
        self.mlp=nn.Sequential(nn.Linear(cfg.n_embd,4*cfg.n_embd,bias=False),nn.SiLU(),nn.Linear(4*cfg.n_embd,cfg.n_embd,bias=False))
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))

class ByteCausalLM(nn.Module):
    def __init__(self,cfg:TextConfig):
        super().__init__()
        self.cfg=cfg
        self.token=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos=nn.Embedding(cfg.seq_len,cfg.n_embd) if cfg.position_encoding == "learned" else None
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        self.head.weight=self.token.weight
        self.gradient_checkpointing=False
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
    def forward(self,idx,targets=None):
        _,t=idx.shape
        if t>self.cfg.seq_len: raise ValueError("sequence too long")
        x=self.token(idx)
        if self.pos is not None:
            p=torch.arange(t,device=idx.device)
            x=x+self.pos(p)[None]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x=torch.utils.checkpoint.checkpoint(block,x,use_reentrant=False)
            else:
                x=block(x)
        logits=self.head(self.ln(x))
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
