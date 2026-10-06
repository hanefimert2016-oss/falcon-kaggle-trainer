#!/usr/bin/env python3
"""Apply a minimal PyTorch/XLA TPU port to Robbyant/lingbot-world-v2.

Designed for Kaggle TPU v5e-8. The patch keeps the CUDA path intact while:
- adds XLA/Pallas attention dispatch,
- removes CUDA-only autocast assumptions on the causal-fast path,
- lets WanI2VCausal accept a torch.device('xla'),
- guards CUDA cache/sync calls,
- makes VAE checkpoint loading CPU-first then transfers to XLA,
- makes XLA random generation avoid CUDA-only Generator assumptions.

This is an experimental inference port, not an upstream-supported configuration.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ATTENTION_XLA = r'''import math
import warnings
import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from torch_xla.experimental.custom_kernel import flash_attention as xla_flash_attention
    XLA_FLASH_ATTN_AVAILABLE = True
except Exception:
    xla_flash_attention = None
    XLA_FLASH_ATTN_AVAILABLE = False

__all__ = ['flash_attention', 'attention']


def _is_xla(x):
    return getattr(getattr(x, 'device', None), 'type', None) == 'xla'


def _sdpa(q, k, v, *, dropout_p=0.0, softmax_scale=None, q_scale=None,
          causal=False, dtype=torch.bfloat16):
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    # PyTorch SDPA default scale is 1/sqrt(d). Preserve a custom scale by
    # folding the ratio into q when the installed PyTorch does not expose
    # the scale kwarg consistently across backends.
    if softmax_scale is not None:
        q = q * (softmax_scale * math.sqrt(q.shape[-1]))
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def _xla_attention(q, k, v, *, dropout_p=0.0, softmax_scale=None,
                   q_scale=None, causal=False, dtype=torch.bfloat16):
    # Pallas flash_attention uses [B, H, S, D]. It does not implement
    # dropout; inference uses dropout_p=0. For unusual shapes/options we
    # deliberately fall back to XLA-lowered SDPA for correctness.
    if dropout_p != 0.0 or not XLA_FLASH_ATTN_AVAILABLE:
        return _sdpa(q, k, v, dropout_p=dropout_p,
                     softmax_scale=softmax_scale, q_scale=q_scale,
                     causal=causal, dtype=dtype)

    out_dtype = q.dtype
    qh = q.transpose(1, 2).to(dtype)
    kh = k.transpose(1, 2).to(dtype)
    vh = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        qh = qh * q_scale
    sm_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qh.shape[-1])
    try:
        out = xla_flash_attention(qh, kh, vh, causal=causal, sm_scale=sm_scale)
    except TypeError:
        # Older/newer torch-xla builds have slightly different kwargs.
        if softmax_scale is not None:
            qh = qh * (softmax_scale * math.sqrt(qh.shape[-1]))
        out = xla_flash_attention(qh, kh, vh, causal=causal)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, version=None,
):
    if _is_xla(q):
        if q_lens is not None or k_lens is not None:
            # Current LingBot inference uses fixed/padded text lengths. For
            # variable-length batches, SDPA is safer until a Pallas varlen
            # kernel is wired in.
            warnings.warn('XLA path ignores q_lens/k_lens padding metadata; use fixed-length inference batches.')
        if window_size != (-1, -1):
            warnings.warn('XLA Pallas path relies on LingBot KV-cache truncation; explicit window_size is ignored.')
        return _xla_attention(q, k, v, dropout_p=dropout_p,
                              softmax_scale=softmax_scale, q_scale=q_scale,
                              causal=causal, dtype=dtype)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    if q.device.type != 'cuda':
        return _sdpa(q, k, v, dropout_p=dropout_p,
                     softmax_scale=softmax_scale, q_scale=q_scale,
                     causal=causal, dtype=dtype)
    assert q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:vv] for u, vv in zip(q, q_lens)]))
    if k_lens is None:
        k = half(k.flatten(0, 1)); v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:vv] for u, vv in zip(k, k_lens)]))
        v = half(torch.cat([u[:vv] for u, vv in zip(v, k_lens)]))
    q = q.to(v.dtype); k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 unavailable; falling back to FA2.')
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32),
            seqused_q=None, seqused_k=None,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        # Should only happen if the module availability changed after import.
        return _sdpa(q.unflatten(0, (b, lq)), k.unflatten(0, (b, lk)),
                     v.unflatten(0, (b, lk)), dropout_p=dropout_p,
                     softmax_scale=softmax_scale, causal=causal, dtype=dtype)
    return x.type(out_dtype)


def attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, fa_version=None,
):
    if _is_xla(q):
        return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens,
                               dropout_p=dropout_p, softmax_scale=softmax_scale,
                               q_scale=q_scale, causal=causal,
                               window_size=window_size,
                               deterministic=deterministic, dtype=dtype,
                               version=fa_version)
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens,
                               dropout_p=dropout_p, softmax_scale=softmax_scale,
                               q_scale=q_scale, causal=causal,
                               window_size=window_size,
                               deterministic=deterministic, dtype=dtype,
                               version=fa_version)
    if q_lens is not None or k_lens is not None:
        warnings.warn('Padding metadata ignored by generic SDPA fallback.')
    return _sdpa(q, k, v, dropout_p=dropout_p,
                 softmax_scale=softmax_scale, q_scale=q_scale,
                 causal=causal, dtype=dtype)
'''


def backup(path: Path) -> None:
    b = path.with_suffix(path.suffix + '.pre_tpu')
    if path.exists() and not b.exists():
        shutil.copy2(path, b)


def replace(path: Path, old: str, new: str, *, required: bool = True) -> bool:
    text = path.read_text()
    if old not in text:
        if required:
            raise RuntimeError(f'Expected pattern not found in {path}: {old[:100]!r}')
        return False
    path.write_text(text.replace(old, new))
    return True


def insert_after(path: Path, needle: str, payload: str) -> None:
    text = path.read_text()
    if payload.strip() in text:
        return
    if needle not in text:
        raise RuntimeError(f'Insertion anchor missing in {path}: {needle!r}')
    path.write_text(text.replace(needle, needle + payload, 1))


def patch(root: Path) -> None:
    # 1) Attention backend dispatch.
    attn = root / 'wan/modules/attention.py'
    backup(attn)
    attn.write_text(ATTENTION_XLA)

    # 2) Core fast/causal blocks: CUDA autocast(float32) was only being used
    # as a precision guard. Explicit .float() is already present, so a no-op
    # context is portable and avoids CUDA-only contexts on TPU.
    for rel in ['wan/modules/model_fast.py', 'wan/modules/model_causal.py']:
        p = root / rel
        if not p.exists():
            continue
        backup(p)
        insert_after(p, 'import math\n', 'from contextib import nullcontext\\n')
        replace(p, "with torch.amp.autocast('cuda', dtype=torch.float32):", 'with nullcontext():', required=False)

    # CUDA-disabled autocast decorators are no-ops for our explicit dtypes;
    # remove them so import does not depend on a CUDA backend.
    for rel in ['wan/modules/model.py', 'wan/modules/model_causal.py', 'wan/distributed/sequence_parallel.py']:
        p = root / rel
        if p.exists():
            backup(p)
            replace(p, "@torch.amp.autocast('cuda', enabled=False)\n", '', required=False)

    # 3) image2video runtime portability.
    p = root / 'wan/image2video.py'
    backup(p)
    helper = r'''


def _portable_device(device_id):
    if isinstance(device_id, torch.device):
        return device_id
    if isinstance(device_id, str):
        return torch.device(device_id)
    return torch.device(f"cuda:{device_id}")


def _portable_autocast(device, dtype):
    from contextlib import nullcontext
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.amp.autocast('cuda', dtype=dtype)
    # TPU parameters/activations are explicitly BF16; XLA handles lowering.
    return nullcontext()


def _portable_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _portable_sync(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'xla':
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()
'''
    insert_after(p, 'import torch\n', helper)
    replace(p, 'self.device = torch.device(f"cuda:{device_id}")', 'self.device = _portable_device(device_id)')
    replace(p, "torch.amp.autocast('cuda', dtype=self.param_dtype)", '_portable_autocast(self.device, self.param_dtype)', required=False)
    replace(p, 'torch.cuda.empty_cache()', '_portable_empty_cache()', required=False)
    replace(p, 'torch.cuda.synchronize()', '_portable_sync(self.device)', required=False)
    old_seed = '''seed_g = torch.Generator(device=self.device)\n        seed_g.manual_seed(seed)'''
    new_seed = '''if self.device.type == 'xla':\n            torch.manual_seed(seed)\n            seed_g = None\n        else:\n            seed_g = torch.Generator(device=self.device)\n            seed_g.manual_seed(seed)'''
    replace(p, old_seed, new_seed, required=False)

    # 4) T5: avoid evaluating torch.cuda.current_device() at import time.
    p = root / 'wan/modules/t5.py'
    backup(p)
    replace(p, 'device=torch.cuda.current_device(),', 'device=None,', required=False)
    replace(p, 'device=torch.cuda.current_device()', 'device=None', required=False)
    replace(p, 'self.dtype = dtype\n        self.device = device',
            "self.dtype = dtype\n        if device is None:\n            device = torch.device('cpu')\n        self.device = device", required=False)

    # 5) VAE: CPU-first load works on CUDA and XLA, then model.to(device).
    p = root / 'wan/modules/vae2_1.py'
    backup(p)
    replace(p, 'import torch.cuda.amp as amp\n', 'from contextlib import nullcontext\n', required=False)
    helper2 = r'''


def _vae_autocast(device, dtype):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.amp.autocast('cuda', dtype=dtype)
    return nullcontext()
'''
    insert_after(p, 'import torch\n', helper2)
    replace(p, 'with amp.autocast(dtype=self.dtype):', 'with _vae_autocast(self.device, self.dtype):', required=False)
    replace(p, 'torch.load(pretrained_path, map_location=device)', "torch.load(pretrained_path, map_location='cpu')", required=False)

    # 6) Keep upstream CUDA requirements untouched except flash_attn becomes
    # optional on TPU. Kaggle script installs normal deps explicitly.
    req = root / 'requirements.txt'
    if req.exists():
        backup(req)
        lines = [ln for ln in req.read_text().splitlines() if ln.strip() != 'flash_attn']
        req.write_text('\n'.join(lines) + '\n')

    marker = root / '.lingbot_tpu_patch_v1'
    marker.write_text('PyTorch/XLA TPU patch v1 applied\n')
    print(f'Patched LingBot TPU runtime at: {root}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo', nargs='?', default='.', help='Path to lingbot-world-v2 clone')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    if not (root / 'wan').exists():
        raise SystemExit(f'Not a LingBot repo: {root}')
    patch(root)


if __name__ == '__main__':
    main()
