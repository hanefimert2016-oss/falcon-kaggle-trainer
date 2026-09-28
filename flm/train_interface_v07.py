#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from flm.models.interface_transformer import InterfaceConfig, InterfaceTransformer
from flm.runtime import select_runtime
from flm.train_v07 import (
    ShuffledStartPool,
    atomic_torch_save,
    eligible_sft_starts,
    make_sft_batch_at_starts,
    make_xy,
    mmap_mask,
    mmap_tokens,
    nonoverlap_starts,
    optimizer_to_device,
    resume_enabled,
    split_train_eval_stream,
)


def resolve_text_root(version: str = "v0.7-dev-text") -> Path:
    configured=os.environ.get("FLM_V07_DATA_ROOT","").strip()
    candidates=[]
    if configured:
        p=Path(configured)
        if (p/"sources.json").is_file():
            candidates.append(p/"sources.json")
        if p.exists():
            candidates.extend(p.rglob("sources.json"))
    kaggle=Path("/kaggle/input")
    if kaggle.exists():
        candidates.extend(kaggle.rglob("sources.json"))
    for manifest in candidates:
        try:
            obj=json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("pipeline_version")==version and int(obj.get("data_revision",0))>=5:
            return manifest.parent
    raise SystemExit(f"could not locate {version} data revision 5+")


def config(vocab_size:int)->InterfaceConfig:
    return InterfaceConfig(
        vocab_size=vocab_size,
        seq_len=int(os.environ.get("FLM_INTERFACE_SEQ","4096")),
        n_layer=int(os.environ.get("FLM_INTERFACE_LAYERS","16")),
        n_head=int(os.environ.get("FLM_INTERFACE_HEADS","12")),
        n_kv_head=int(os.environ.get("FLM_INTERFACE_KV_HEADS","4")),
        n_embd=int(os.environ.get("FLM_INTERFACE_EMBD","768")),
        hidden_mult=float(os.environ.get("FLM_INTERFACE_HIDDEN_MULT",str(8.0/3.0))),
        rope_theta=float(os.environ.get("FLM_INTERFACE_ROPE_THETA","10000")),
    )


def _lr(step:int,total:int,peak:float,floor:float,warmup:int)->float:
    if step < warmup:
        return peak*(step+1)/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup-1)
    return floor+0.5*(peak-floor)*(1+math.cos(math.pi*min(1.0,max(0.0,p))))


@torch.no_grad()
def eval_pretrain(model,data,seq,device,rng,batches=16):
    model.eval()
    vals=[]
    for _ in range(max(1,batches)):
        high=len(data)-seq-2
        s=int(rng.integers(0,high))
        x,y=make_xy(data,np.asarray([s],dtype=np.int64),seq,device)
        _,loss=model(x,y)
        vals.append(float(loss.detach()))
    model.train()
    return sum(vals)/len(vals)


@torch.no_grad()
def eval_sft(model,data,mask,seq,device,rng,batches=16):
    model.eval()
    starts=eligible_sft_starts(mask,seq,min_supervised=8)
    vals=[]
    for _ in range(max(1,batches)):
        s=starts[int(rng.integers(0,len(starts)))]
        x,y,_=make_sft_batch_at_starts(data,mask,[s],seq,device)
        _,loss=model(x,y)
        vals.append(float(loss.detach()))
    model.train()
    return sum(vals)/len(vals)


def save_resume(path,model,opt,scaler,pool,stage,next_step,config,extra=None):
    atomic_torch_save({
        "format":"flm-interface-v07-resume",
        "stage":stage,
        "model":model.state_dict(),
        "optimizer":opt.state_dict(),
        "scaler":scaler.state_dict() if scaler.is_enabled() else None,
        "sampler":pool.state_dict(),
        "next_step":int(next_step),
        "config":config.__dict__,
        **(extra or {}),
    },path)


def train_pretrain_stage(
    model,
    *,
    name,
    data,
    cfg,
    runtime,
    out,
    train_seq,
    batch,
    accum,
    epochs,
    peak_lr,
    floor_lr,
    seed,
):
    train,held=split_train_eval_stream(data,train_seq,eval_fraction=0.01)
    starts=nonoverlap_starts(len(train),train_seq)
    pool=ShuffledStartPool(starts,seed)
    per_step=batch*accum
    steps=max(1,math.ceil(len(starts)*epochs/max(1,per_step)))
    warmup=max(20,min(800,steps//20))
    opt=torch.optim.AdamW(model.parameters(),lr=peak_lr,betas=(0.9,0.95),weight_decay=0.1)
    scaler=torch.amp.GradScaler("cuda",enabled=runtime.kind=="gpu")
    wrapped=model
    if runtime.kind=="gpu" and torch.cuda.device_count()>1 and batch>=2:
        wrapped=torch.nn.DataParallel(model)
    stage_dir=out/"stages"
    stage_dir.mkdir(parents=True,exist_ok=True)
    resume=stage_dir/f"{name}_resume.pt"
    best_path=stage_dir/f"{name}_best.pt"
    final=stage_dir/f"{name}.pt"
    start_step=0
    tokens_seen=0
    best=float("inf")
    eval_rng=np.random.default_rng(seed+10000)
    ckpt_interval=int(os.environ.get("FLM_INTERFACE_CKPT_INTERVAL","500"))
    eval_interval=int(os.environ.get("FLM_INTERFACE_EVAL_INTERVAL","500"))

    if resume_enabled() and final.is_file():
        ck=torch.load(final,map_location=runtime.device,weights_only=False)
        if ck.get("config")==cfg.__dict__ and int(ck.get("steps",0))>=steps:
            model.load_state_dict(ck["model"])
            return ck["metrics"]

    if resume_enabled() and resume.is_file():
        ck=torch.load(resume,map_location=runtime.device,weights_only=False)
        if ck.get("config")!=cfg.__dict__ or ck.get("stage")!=name:
            raise RuntimeError(f"{name} resume mismatch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        optimizer_to_device(opt,runtime.device)
        if scaler.is_enabled() and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        pool.load_state_dict(ck["sampler"])
        start_step=int(ck["next_step"])
        tokens_seen=int(ck.get("tokens_seen",0))
        best=float(ck.get("best_eval",best))
        print(f"INTERFACE_RESUME stage={name} step={start_step}/{steps}",flush=True)

    started=time.time()
    for step in range(start_step,steps):
        lr=_lr(step,steps,peak_lr,floor_lr,warmup)
        for group in opt.param_groups:
            group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        loss_sum=0.0
        for _ in range(accum):
            sel=pool.take(batch)
            x,y=make_xy(train,sel,train_seq,runtime.device)
            with runtime.autocast():
                _,loss=wrapped(x,y)
                if loss.ndim:
                    loss=loss.mean()
                loss=loss/accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite interface loss stage={name} step={step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_sum+=float(loss.detach())
            tokens_seen+=batch*train_seq
        if scaler.is_enabled():
            scaler.unscale_(opt)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"non-finite interface gradient stage={name} step={step}")
        runtime.optimizer_step(opt,scaler)

        if (step+1)%eval_interval==0 or step==steps-1:
            ev=eval_pretrain(model,held,train_seq,runtime.device,eval_rng,
                             int(os.environ.get("FLM_INTERFACE_EVAL_BATCHES","16")))
            if ev < best:
                best=ev
                atomic_torch_save({
                    "model":model.state_dict(),
                    "config":cfg.__dict__,
                    "eval_loss":ev,
                    "step":step+1,
                },best_path)
            print(
                f"interface {name} step={step+1}/{steps} loss={loss_sum:.4f} "
                f"eval={ev:.4f} best={best:.4f} tokens={tokens_seen}",
                flush=True,
            )
        if ckpt_interval>0 and (step+1)%ckpt_interval==0:
            save_resume(
                resume,model,opt,scaler,pool,name,step+1,cfg,
                {"tokens_seen":tokens_seen,"best_eval":best},
            )

    if best_path.is_file():
        best_ck=torch.load(best_path,map_location=runtime.device,weights_only=False)
        model.load_state_dict(best_ck["model"])
    ev=eval_pretrain(model,held,train_seq,runtime.device,eval_rng,
                     int(os.environ.get("FLM_INTERFACE_EVAL_BATCHES","16")))
    metrics={
        "stage":name,
        "steps":steps,
        "epochs":epochs,
        "training_seq_len":train_seq,
        "tokens_seen":tokens_seen,
        "eval_loss":ev,
        "best_eval_loss":best,
        "elapsed_s":time.time()-started,
    }
    atomic_torch_save({
        "format":"flm-interface-v07-stage",
        "stage":name,
        "model":model.state_dict(),
        "config":cfg.__dict__,
        "steps":steps,
        "metrics":metrics,
    },final)
    resume.unlink(missing_ok=True)
    return metrics


def train_mixed_sft(
    model,
    *,
    main_data,
    main_mask,
    coder_data,
    coder_mask,
    cfg,
    runtime,
    out,
    seq,
    batch,
    accum,
    epochs,
):
    md,me=split_train_eval_stream(main_data,seq,eval_fraction=0.02)
    cd,ce=split_train_eval_stream(coder_data,seq,eval_fraction=0.02)
    mm=main_mask[:len(md)]; mem=main_mask[len(md):]
    cm=coder_mask[:len(cd)]; cem=coder_mask[len(cd):]
    ms=eligible_sft_starts(mm,seq,min_supervised=8)
    cs=eligible_sft_starts(cm,seq,min_supervised=8)
    mp=ShuffledStartPool(ms,8111)
    cp=ShuffledStartPool(cs,8222)
    # 2:1 main/general+semantic to coding/tool traces.
    virtual=len(ms)+max(1,len(cs))
    steps=max(1,math.ceil(virtual*epochs/max(1,batch*accum)))
    peak=float(os.environ.get("FLM_INTERFACE_SFT_LR","6e-5"))
    floor=float(os.environ.get("FLM_INTERFACE_SFT_MIN_LR","6e-6"))
    warmup=max(20,min(400,steps//20))
    opt=torch.optim.AdamW(model.parameters(),lr=peak,betas=(0.9,0.95),weight_decay=0.05)
    scaler=torch.amp.GradScaler("cuda",enabled=runtime.kind=="gpu")
    wrapped=model
    if runtime.kind=="gpu" and torch.cuda.device_count()>1 and batch>=2:
        wrapped=torch.nn.DataParallel(model)
    resume=out/"stages"/"mixed_sft_resume.pt"
    best_path=out/"stages"/"mixed_sft_best.pt"
    final=out/"stages"/"mixed_sft.pt"
    start_step=0
    best=float("inf")
    main_seen=coder_seen=0
    eval_rng=np.random.default_rng(8333)
    ckpt_interval=int(os.environ.get("FLM_INTERFACE_CKPT_INTERVAL","500"))
    eval_interval=int(os.environ.get("FLM_INTERFACE_SFT_EVAL_INTERVAL","250"))

    if resume_enabled() and final.is_file():
        ck=torch.load(final,map_location=runtime.device,weights_only=False)
        if ck.get("config")==cfg.__dict__ and int(ck.get("steps",0))>=steps:
            model.load_state_dict(ck["model"])
            return ck["metrics"]

    if resume_enabled() and resume.is_file():
        ck=torch.load(resume,map_location=runtime.device,weights_only=False)
        if ck.get("config")!=cfg.__dict__:
            raise RuntimeError("mixed SFT resume config mismatch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"]); optimizer_to_device(opt,runtime.device)
        if scaler.is_enabled() and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        mp.load_state_dict(ck["main_sampler"]); cp.load_state_dict(ck["coder_sampler"])
        start_step=int(ck["next_step"])
        main_seen=int(ck.get("main_supervised",0))
        coder_seen=int(ck.get("coder_supervised",0))
        best=float(ck.get("best_eval",best))

    started=time.time()
    for step in range(start_step,steps):
        lr=_lr(step,steps,peak,floor,warmup)
        for group in opt.param_groups: group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        total=0.0
        for micro in range(accum):
            use_coder=((step*accum+micro)%3)==2
            if use_coder:
                starts=cp.take(batch)
                x,y,sup=make_sft_batch_at_starts(cd,cm,starts,seq,runtime.device)
                coder_seen+=sup
            else:
                starts=mp.take(batch)
                x,y,sup=make_sft_batch_at_starts(md,mm,starts,seq,runtime.device)
                main_seen+=sup
            with runtime.autocast():
                _,loss=wrapped(x,y)
                if loss.ndim: loss=loss.mean()
                loss=loss/accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite mixed SFT loss step={step}")
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            total+=float(loss.detach())
        if scaler.is_enabled(): scaler.unscale_(opt)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"non-finite mixed SFT gradient step={step}")
        runtime.optimizer_step(opt,scaler)

        if (step+1)%eval_interval==0 or step==steps-1:
            em=eval_sft(model,me,mem,seq,runtime.device,eval_rng,8)
            ec=eval_sft(model,ce,cem,seq,runtime.device,eval_rng,8)
            score=0.65*em+0.35*ec
            if score < best:
                best=score
                atomic_torch_save({
                    "model":model.state_dict(),
                    "config":cfg.__dict__,
                    "score":score,
                    "step":step+1,
                },best_path)
            print(
                f"interface mixed_sft step={step+1}/{steps} loss={total:.4f} "
                f"main_eval={em:.4f} coder_eval={ec:.4f} score={score:.4f}",
                flush=True,
            )
        if ckpt_interval>0 and (step+1)%ckpt_interval==0:
            atomic_torch_save({
                "format":"flm-interface-v07-resume",
                "stage":"mixed_sft",
                "model":model.state_dict(),
                "optimizer":opt.state_dict(),
                "scaler":scaler.state_dict() if scaler.is_enabled() else None,
                "main_sampler":mp.state_dict(),
                "coder_sampler":cp.state_dict(),
                "next_step":step+1,
                "main_supervised":main_seen,
                "coder_supervised":coder_seen,
                "best_eval":best,
                "config":cfg.__dict__,
            },resume)

    if best_path.is_file():
        best_ck=torch.load(best_path,map_location=runtime.device,weights_only=False)
        model.load_state_dict(best_ck["model"])
    em=eval_sft(model,me,mem,seq,runtime.device,eval_rng,12)
    ec=eval_sft(model,ce,cem,seq,runtime.device,eval_rng,12)
    metrics={
        "stage":"mixed_sft",
        "steps":steps,
        "training_seq_len":seq,
        "main_supervised_seen":main_seen,
        "coder_supervised_seen":coder_seen,
        "main_eval_loss":em,
        "coder_eval_loss":ec,
        "combined_eval_loss":0.65*em+0.35*ec,
        "best_combined_eval_loss":best,
        "elapsed_s":time.time()-started,
    }
    atomic_torch_save({
        "format":"flm-interface-v07-stage",
        "stage":"mixed_sft",
        "model":model.state_dict(),
        "config":cfg.__dict__,
        "steps":steps,
        "metrics":metrics,
    },final)
    resume.unlink(missing_ok=True)
    return metrics


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    root=resolve_text_root(os.environ.get("FLM_V07_TEXT_VERSION","v0.7-dev-text"))
    manifest=json.loads((root/"sources.json").read_text(encoding="utf-8"))
    if int(manifest.get("data_revision",0))<5:
        raise RuntimeError("single-transformer FLM requires text data revision 5+")
    semantic_rows=int(((manifest.get("stats") or {}).get("semantic_interface_sft") or {}).get("rows",0))
    if semantic_rows<200_000:
        raise RuntimeError(f"semantic interface curriculum too small: {semantic_rows}")

    tok=Tokenizer.from_file(str(root/"tokenizer.json"))
    vocab=tok.get_vocab_size()
    for special in (
        "<|semantic_ir|>","<|semantic_end|>","<|core_result|>","<|core_end|>",
        "<|tool_call|>","<|tool_result|>","<|final|>",
    ):
        if tok.token_to_id(special) is None:
            raise RuntimeError(f"tokenizer missing {special}")

    cfg=config(vocab)
    runtime=select_runtime(os.environ.get("FLM_ACCELERATOR","gpu"))
    out=Path(os.environ.get("FLM_INTERFACE_OUTPUT_ROOT","/kaggle/working/flm-v0.7-semantic"))
    (out/"stages").mkdir(parents=True,exist_ok=True)
    model=InterfaceTransformer(cfg).to(runtime.device)
    model.gradient_checkpointing=os.environ.get(
        "FLM_INTERFACE_GRADIENT_CHECKPOINTING","1"
    ).lower() not in {"0","false","no","off"}

    if args.smoke:
        total=model.parameter_count()
        print("FLM_INTERFACE_SMOKE="+json.dumps({
            "parameters":total,"config":cfg.__dict__,"semantic_rows":semantic_rows
        }))
        return 0

    batch=int(os.environ.get("FLM_INTERFACE_BATCH","2"))
    accum=int(os.environ.get("FLM_INTERFACE_ACCUM","8"))

    main=mmap_tokens(root/"main_train.u16")
    coder=mmap_tokens(root/"coder_train.u16")
    main_sft=mmap_tokens(root/"main_sft_tokens.u16")
    main_mask=mmap_mask(root/"main_sft_mask.u8")
    coder_sft=mmap_tokens(root/"coder_sft_tokens.u16")
    coder_mask=mmap_mask(root/"coder_sft_mask.u8")

    summary={
        "format":"flm-semantic-v07-single-transformer",
        "architecture":"InterfaceTransformer + training-free FLM Core",
        "transformer_count":1,
        "parameters":model.parameter_count(),
        "gpu_names":runtime.gpu_names,
        "data_revision":int(manifest.get("data_revision",0)),
        "semantic_interface_rows":semantic_rows,
        "stages":{},
    }

    summary["stages"]["general_pretrain"]=train_pretrain_stage(
        model,name="general_pretrain",data=main,cfg=cfg,runtime=runtime,out=out,
        train_seq=int(os.environ.get("FLM_INTERFACE_GENERAL_SEQ","1024")),
        batch=batch,accum=accum,
        epochs=float(os.environ.get("FLM_INTERFACE_GENERAL_EPOCHS","1.0")),
        peak_lr=float(os.environ.get("FLM_INTERFACE_GENERAL_LR","2.5e-4")),
        floor_lr=float(os.environ.get("FLM_INTERFACE_GENERAL_MIN_LR","2.5e-5")),
        seed=7001,
    )
    summary["stages"]["code_pretrain"]=train_pretrain_stage(
        model,name="code_pretrain",data=coder,cfg=cfg,runtime=runtime,out=out,
        train_seq=int(os.environ.get("FLM_INTERFACE_CODE_SEQ","2048")),
        batch=batch,accum=accum,
        epochs=float(os.environ.get("FLM_INTERFACE_CODE_EPOCHS","1.0")),
        peak_lr=float(os.environ.get("FLM_INTERFACE_CODE_LR","1.2e-4")),
        floor_lr=float(os.environ.get("FLM_INTERFACE_CODE_MIN_LR","1.2e-5")),
        seed=7002,
    )
    # Replay a small slice of general text after code continued-pretraining.
    # This reduces catastrophic forgetting while keeping one single Transformer.
    summary["stages"]["general_refresh"]=train_pretrain_stage(
        model,name="general_refresh",data=main,cfg=cfg,runtime=runtime,out=out,
        train_seq=int(os.environ.get("FLM_INTERFACE_REFRESH_SEQ","2048")),
        batch=batch,accum=accum,
        epochs=float(os.environ.get("FLM_INTERFACE_REFRESH_EPOCHS","0.12")),
        peak_lr=float(os.environ.get("FLM_INTERFACE_REFRESH_LR","6e-5")),
        floor_lr=float(os.environ.get("FLM_INTERFACE_REFRESH_MIN_LR","6e-6")),
        seed=7003,
    )
    summary["stages"]["mixed_sft"]=train_mixed_sft(
        model,
        main_data=main_sft,main_mask=main_mask,
        coder_data=coder_sft,coder_mask=coder_mask,
        cfg=cfg,runtime=runtime,out=out,
        seq=int(os.environ.get("FLM_INTERFACE_SFT_SEQ","4096")),
        batch=batch,accum=accum,
        epochs=float(os.environ.get("FLM_INTERFACE_SFT_EPOCHS","1.0")),
    )

    final_dir=out/"interface"
    final_dir.mkdir(parents=True,exist_ok=True)
    atomic_torch_save({
        "format":"flm-semantic-v07-interface",
        "model":model.state_dict(),
        "config":cfg.__dict__,
        "result":summary,
    },final_dir/"checkpoint.pt")
    shutil.copy2(root/"tokenizer.json",final_dir/"tokenizer.json")
    (final_dir/"metrics.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print("FLM_INTERFACE_RESULT="+json.dumps(summary,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
