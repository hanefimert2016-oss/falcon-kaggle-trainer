from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F

from flm.models.core_memory import CoreMemoryBank


@dataclass
class InterfaceConfig:
    vocab_size: int = 32768
    seq_len: int = 4096
    n_layer: int = 16
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    hidden_mult: float = 8.0 / 3.0
    rope_theta: float = 10000.0
    dropout: float = 0.0
    core_memory: bool = True
    core_memory_order: int = 4
    core_memory_logit_scale: float = 1.5
    core_memory_residual_scale: float = 0.08


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim))
        self.eps=eps

    def forward(self, x):
        dtype=x.dtype
        x=x.float()
        x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        return (x*self.weight.float()).to(dtype)


def _round_hidden(value: float, multiple: int = 256) -> int:
    return int(math.ceil(value / multiple) * multiple)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even")
        inv=1.0/(theta ** (torch.arange(0,dim,2).float()/dim))
        self.register_buffer("inv_freq",inv,persistent=False)

    def apply(self, q, k):
        t=q.size(-2)
        pos=torch.arange(t,device=q.device,dtype=self.inv_freq.dtype)
        freq=torch.outer(pos,self.inv_freq)
        cos=freq.cos().to(q.dtype)[None,None,:,:]
        sin=freq.sin().to(q.dtype)[None,None,:,:]

        def rot(x):
            xe=x[...,0::2]
            xo=x[...,1::2]
            return torch.stack(
                (xe*cos-xo*sin,xe*sin+xo*cos),dim=-1
            ).flatten(-2)
        return rot(q),rot(k)


class GQAAttention(nn.Module):
    def __init__(self,cfg:InterfaceConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must divide n_head")
        if cfg.n_head % cfg.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head=cfg.n_head
        self.n_kv_head=cfg.n_kv_head
        self.head_dim=cfg.n_embd//cfg.n_head
        self.group=cfg.n_head//cfg.n_kv_head
        self.q_proj=nn.Linear(cfg.n_embd,cfg.n_head*self.head_dim,bias=False)
        self.k_proj=nn.Linear(cfg.n_embd,cfg.n_kv_head*self.head_dim,bias=False)
        self.v_proj=nn.Linear(cfg.n_embd,cfg.n_kv_head*self.head_dim,bias=False)
        self.o_proj=nn.Linear(cfg.n_embd,cfg.n_embd,bias=False)
        self.rope=RotaryEmbedding(self.head_dim,cfg.rope_theta)
        self.dropout=cfg.dropout

    def forward(self,x):
        b,t,_=x.shape
        q=self.q_proj(x).view(b,t,self.n_head,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(b,t,self.n_kv_head,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(b,t,self.n_kv_head,self.head_dim).transpose(1,2)
        if self.group>1:
            k=k.repeat_interleave(self.group,dim=1)
            v=v.repeat_interleave(self.group,dim=1)
        q,k=self.rope.apply(q,k)
        y=F.scaled_dot_product_attention(
            q,k,v,is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,-1))


class SwiGLU(nn.Module):
    def __init__(self,cfg:InterfaceConfig):
        super().__init__()
        hidden=_round_hidden(cfg.hidden_mult*cfg.n_embd)
        self.gate=nn.Linear(cfg.n_embd,hidden,bias=False)
        self.up=nn.Linear(cfg.n_embd,hidden,bias=False)
        self.down=nn.Linear(hidden,cfg.n_embd,bias=False)

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))


class InterfaceBlock(nn.Module):
    def __init__(self,cfg:InterfaceConfig):
        super().__init__()
        self.attn_norm=RMSNorm(cfg.n_embd)
        self.attn=GQAAttention(cfg)
        self.ffn_norm=RMSNorm(cfg.n_embd)
        self.ffn=SwiGLU(cfg)

    def forward(self,x):
        x=x+self.attn(self.attn_norm(x))
        return x+self.ffn(self.ffn_norm(x))


class InterfaceTransformer(nn.Module):
    """The only trainable Transformer in Semantic FLM.

    It handles language/code <-> structured protocol. Reasoning, memory,
    planning and verification live outside this network in FLM Core.
    """

    def __init__(self,cfg:InterfaceConfig):
        super().__init__()
        self.cfg=cfg
        self.token=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.blocks=nn.ModuleList([InterfaceBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm=RMSNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        self.head.weight=self.token.weight
        self.gradient_checkpointing=False
        self.core_memory: CoreMemoryBank|None=None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)

    def attach_core_memory(self,bank:CoreMemoryBank|None):
        if bank is not None:
            if bank.vocab_size!=self.cfg.vocab_size:
                raise RuntimeError(
                    f"CoreMemory vocab mismatch: {bank.vocab_size}!={self.cfg.vocab_size}"
                )
            if bank.order!=self.cfg.core_memory_order:
                raise RuntimeError(
                    f"CoreMemory order mismatch: {bank.order}!={self.cfg.core_memory_order}"
                )
        self.core_memory=bank
        return self

    def forward(self,idx,targets=None):
        _,t=idx.shape
        if t>self.cfg.seq_len:
            raise ValueError(f"sequence too long: {t}>{self.cfg.seq_len}")
        x=self.token(idx)
        if self.core_memory is not None and self.cfg.core_memory_residual_scale:
            x=x+self.cfg.core_memory_residual_scale*self.core_memory.residual(idx,self.token)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x=torch.utils.checkpoint.checkpoint(block,x,use_reentrant=False)
            else:
                x=block(x)
        logits=self.head(self.norm(x))
        if self.core_memory is not None:
            logits=self.core_memory.apply_logits(
                idx,logits,self.cfg.core_memory_logit_scale
            )
        loss=None
        if targets is not None:
            loss=F.cross_entropy(
                logits.reshape(-1,logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits,loss

    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())
