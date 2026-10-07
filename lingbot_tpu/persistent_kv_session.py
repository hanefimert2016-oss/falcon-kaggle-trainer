#!/usr/bin/env python3
"""Experimental persistent-KV LingBot causal streaming.

Unlike live_session.py, this path does NOT call pipe.generate() per control
update. It keeps the causal self-attention KV cache, text cross-attention
cache, scheduler and temporal current_start alive across WASD chunks.

It mirrors the released _generate_causal_fast inner loop for one 2-latent
chunk at a time. This is intentionally isolated until a real TPU run can
compare it against the conservative model-resident I2V feedback path.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange

from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)

from wasd_camera import WasdCamera


def _autocast(device, dtype):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _sync(device):
    device = torch.device(device)
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class PersistentKVCausalSession:
    """One persistent causal-fast stream with 2 latent frames per control step."""

    def __init__(
        self,
        pipe,
        cfg,
        *,
        initial_image,
        prompt: str,
        intrinsics_source: str | Path,
        max_area: int = 256 * 448,
        chunk_size: int = 2,
        timesteps_index=(0, 179, 358, 679),
        shift: float | None = None,
        seed: int = 42,
        max_sequence_length: int = 512,
        max_attention_size: int | None = None,
        move_per_frame: float = 0.08,
        yaw_per_frame_deg: float = 2.0,
    ):
        if chunk_size != 2:
            raise ValueError("Persistent interactive mode currently requires chunk_size=2")
        self.pipe = pipe
        self.cfg = cfg
        self.device = torch.device(pipe.device)
        self.prompt = str(prompt)
        self.max_area = int(max_area)
        self.chunk_size = int(chunk_size)
        self.max_sequence_length = int(max_sequence_length)
        self.max_attention_size_override = max_attention_size
        self.seed = int(seed)
        self.step_index = 0
        self.camera = WasdCamera(
            move_per_frame=move_per_frame,
            yaw_per_frame_deg=yaw_per_frame_deg,
        )

        # 5 visible frames -> 2 VAE temporal latents for stride 4.
        self.frame_num = 5
        self.lat_f = 2

        img = TF.to_tensor(initial_image.convert("RGB")).sub_(0.5).div_(0.5)
        self.input_h, self.input_w = img.shape[1:]
        aspect_ratio = self.input_h / self.input_w

        self.lat_h = round(
            np.sqrt(self.max_area * aspect_ratio)
            // pipe.vae_stride[1]
            // pipe.patch_size[1]
            * pipe.patch_size[1]
        )
        self.lat_w = round(
            np.sqrt(self.max_area / aspect_ratio)
            // pipe.vae_stride[2]
            // pipe.patch_size[2]
            * pipe.patch_size[2]
        )
        self.h = self.lat_h * pipe.vae_stride[1]
        self.w = self.lat_w * pipe.vae_stride[2]
        self.frame_seqlen = int(
            self.lat_h * self.lat_w
            // (pipe.patch_size[1] * pipe.patch_size[2])
        )
        self.max_seq_len = int(
            math.ceil(
                (self.chunk_size * self.frame_seqlen) / pipe.sp_size
            ) * pipe.sp_size
        )

        self.current_start = 0
        self._cross_attn_initialized = False

        # Keep a deterministic XLA/CUDA RNG stream without constructing an XLA
        # torch.Generator (unsupported in several torch-xla versions).
        if self.device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(self.seed, self.device)
            self.generator = None
        else:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)

        pipe.scheduler.set_timesteps(
            pipe.num_train_timesteps,
            shift=float(shift if shift is not None else cfg.sample_shift),
        )
        self.timesteps = pipe.scheduler.timesteps[list(timesteps_index)]

        self.context = self._prepare_context()
        self.K = self._prepare_intrinsics(intrinsics_source)
        self.first_condition, self.blank_condition = self._prepare_conditions(
            img.to(self.device)
        )
        self._initialize_caches()
        _sync(self.device)

    def _randn(self, shape, dtype=torch.float32):
        return torch.randn(
            shape,
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def _prepare_context(self):
        key = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        if key in self.pipe._t5_cache:
            return self.pipe._t5_cache[key]

        if self.pipe.t5_cpu:
            context = self.pipe.text_encoder([self.prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
        else:
            self.pipe.text_encoder.model.to(self.device)
            context = self.pipe.text_encoder([self.prompt], self.device)
        self.pipe._t5_cache[key] = context
        return context

    def _prepare_intrinsics(self, source):
        Ks = torch.from_numpy(np.load(source)).float()
        Ks = get_Ks_transformed(
            Ks,
            height_org=480,
            width_org=832,
            height_resize=self.h,
            width_resize=self.w,
            height_final=self.h,
            width_final=self.w,
        )
        return Ks[0].to(self.device)

    def _condition_mask(self, first: bool):
        if not first:
            return torch.zeros(
                4, self.chunk_size, self.lat_h, self.lat_w,
                device=self.device,
                dtype=self.pipe.param_dtype,
            )
        msk = torch.ones(
            1, self.frame_num, self.lat_h, self.lat_w,
            device=self.device,
        )
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(
            1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w
        ).transpose(1, 2)[0]
        return msk.to(self.pipe.param_dtype)

    def _prepare_conditions(self, img):
        resized = torch.nn.functional.interpolate(
            img[None].cpu(),
            size=(self.h, self.w),
            mode="bicubic",
        ).transpose(0, 1)

        first_video = torch.concat(
            [
                resized,
                torch.zeros(3, self.frame_num - 1, self.h, self.w),
            ],
            dim=1,
        ).to(self.device)
        first_vae = self.pipe.vae.encode([first_video])[0]
        first = torch.concat([self._condition_mask(True), first_vae])

        # Upstream long-video I2V conditions future chunks on the VAE encoding
        # of zero-valued future frames. Precompute that stationary condition
        # once instead of re-encoding on every WASD update.
        blank_video = torch.zeros(
            3, self.frame_num, self.h, self.w,
            device=self.device,
            dtype=img.dtype,
        )
        blank_vae = self.pipe.vae.encode([blank_video])[0]
        blank = torch.concat([self._condition_mask(False), blank_vae])
        return first, blank

    def _initialize_caches(self):
        model_args = self.pipe.model.config
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.pipe.sp_size

        if self.pipe.local_attn_size > -1:
            kv_size = self.frame_seqlen * self.pipe.local_attn_size
        else:
            # Interactive mode is intended to use the released local-attention
            # window. A finite fallback still leaves room for many chunks.
            kv_size = self.frame_seqlen * 128
        self.kv_size = int(kv_size)

        self.self_kv_cache = self.pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[
                1,
                self.kv_size,
                local_num_heads,
                head_dim,
            ],
            dtype=self.pipe.pipe_dtype,
            device=self.device,
        )
        self.cross_kv_cache = self.pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[
                1,
                self.max_sequence_length,
                model_args.num_heads,
                head_dim,
            ],
            dtype=self.pipe.pipe_dtype,
            device=self.device,
        )

    def _camera_condition(self, keys, mouse_dx=0.0, mouse_dy=0.0):
        poses = self.camera.trajectory(
            keys,
            frame_num=5,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
        )
        len_c2ws = len(poses)
        latent_pose_count = int((len_c2ws - 1) // 4) + 1
        latent_pose_count -= latent_pose_count % self.chunk_size
        c2ws = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=poses[:, :3, :3],
            src_trans_vec=poses[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, latent_pose_count),
        )
        c2ws = compute_relative_poses(c2ws, framewise=True).to(self.device)
        Ks = self.K.repeat(len(c2ws), 1)

        plucker = get_plucker_embeddings(c2ws, Ks, self.h, self.w)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(self.h // self.lat_h),
            c2=int(self.w // self.lat_w),
        )
        plucker = plucker[None, ...]
        plucker = rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=self.lat_f,
            h=self.lat_h,
            w=self.lat_w,
        ).to(self.pipe.param_dtype)
        return plucker

    def step(self, keys=(), *, mouse_dx=0.0, mouse_dy=0.0):
        keys = tuple(sorted(str(k).lower() for k in keys))
        current_latent = self._randn(
            (16, self.chunk_size, self.lat_h, self.lat_w),
            dtype=torch.float32,
        )
        current_condition = (
            self.first_condition if self.step_index == 0 else self.blank_condition
        )
        camera_condition = self._camera_condition(
            keys, mouse_dx=mouse_dx, mouse_dy=mouse_dy
        )

        kwargs = {
            "context": [self.context[0]],
            "seq_len": self.max_seq_len,
            "y": [current_condition],
            "dit_cond_dict": {
                "c2ws_plucker_emb": camera_condition.chunk(1, dim=0),
            },
            "kv_cache": self.self_kv_cache,
            "crossattn_cache": self.cross_kv_cache,
            "current_start": self.current_start,
            "max_attention_size": (
                self.kv_size
                if self.max_attention_size_override is None
                else self.max_attention_size_override
            ),
            "frame_seqlen": self.frame_seqlen,
        }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.pipe.model, "no_sync", noop_no_sync)
        t0 = time.perf_counter()

        with _autocast(self.device, self.pipe.param_dtype), torch.no_grad(), no_sync():
            for timestep_idx, current_timestep in enumerate(self.timesteps):
                timestep = torch.stack([current_timestep]).to(self.device)
                noise_pred = self.pipe.model(
                    x=[current_latent],
                    t=timestep,
                    cross_attn_first_call=not self._cross_attn_initialized,
                    **kwargs,
                )[0]
                self._cross_attn_initialized = True

                x0 = self.pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=current_latent,
                    timestep=current_timestep,
                    scheduler=self.pipe.scheduler,
                )

                if timestep_idx < len(self.timesteps) - 1:
                    next_timestep = self.timesteps[timestep_idx + 1]
                    current_latent = self.pipe.scheduler.add_noise(
                        x0,
                        self._randn(x0.shape, dtype=x0.dtype),
                        next_timestep,
                    )

            # Commit clean x0 K/V for this temporal position, exactly like the
            # released causal-fast loop does after every chunk.
            zero_t = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
            self.pipe.model(
                x=[x0],
                t=zero_t,
                cross_attn_first_call=False,
                **kwargs,
            )

        self.current_start += self.chunk_size * self.frame_seqlen
        self.step_index += 1

        # Experimental decode: decode this 2-latent chunk independently.
        # The model temporal state is persistent; a future streaming-VAE
        # decoder can eliminate the remaining chunk-boundary decode overlap.
        video = self.pipe.vae.decode([x0])[0]
        _sync(self.device)
        elapsed = time.perf_counter() - t0

        # One 2-latent independent decode produces 5 visible frames. Treat the
        # first as overlap/context and expose four new display frames.
        display = video[:, 1:]
        fps = int(display.shape[1]) / elapsed
        stats = {
            "seconds": elapsed,
            "generated_frames": int(display.shape[1]),
            "fps": fps,
            "max_area": self.max_area,
            "step": self.step_index,
            "current_start": self.current_start,
            "keys": keys,
        }
        return display, stats


__all__ = ["PersistentKVCausalSession"]
