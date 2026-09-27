#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from datasets import load_dataset
from tokenizers import Tokenizer

from flm.data.prepare_v07 import (
    SOURCES,
    clean_text,
    compact_json,
    copy_prefix,
    encode_chat_sft,
    encode_coder_pretrain,
    encode_main,
    prepare_coder_raw,
    prepare_coder_sft,
    prepare_main_raw,
    prepare_main_sft,
    train_tokenizer,
)

DEV_SOURCES = {
    "main_sft_tr_knowledge": {
        "repo": "Uunan/turkish-knowledge-sft",
        "license": "apache-2.0",
        "role": "large Turkish knowledge/instruction SFT",
    },
    "coder_tools_100k": {
        "repo": "stindardlogic/tool-calling-english-100k",
        "license": "apache-2.0",
        "role": "100k full-cycle tool-calling conversations",
    },
}


def normalize_messages(messages):
    out=[]
    for m in messages or []:
        if not isinstance(m,dict):
            continue
        role=str(m.get("role") or "").strip().lower()
        if role not in {"system","user","assistant","tool"}:
            continue
        content=m.get("content")
        if role=="assistant" and m.get("tool_calls"):
            calls=[]
            for tc in m.get("tool_calls") or []:
                fn=tc.get("function") if isinstance(tc,dict) else None
                if not isinstance(fn,dict):
                    continue
                name=str(fn.get("name") or "")
                args=fn.get("arguments",{})
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except Exception: args={"raw":args}
                if name and isinstance(args,dict):
                    calls.append("<|tool_call|>"+compact_json({"name":name,"arguments":args})+"<|tool_end|>")
            text="\n".join(calls)
            if content:
                text=(clean_text(content)+"\n"+text).strip()
            content=text
        elif role=="tool":
            content=clean_text(content)
            if content and not content.startswith("<|tool_result|>"):
                name=str(m.get("name") or "tool")
                content="<|tool_result|>"+compact_json({"name":name,"result":content})+"<|tool_end|>"
        else:
            content=clean_text(content)
        if content:
            out.append({"role":role,"content":content})
    return out


def append_turkish_knowledge(path: Path, limit: int) -> dict:
    added=0
    ds=load_dataset(DEV_SOURCES["main_sft_tr_knowledge"]["repo"],split="train",streaming=True)
    with path.open("a",encoding="utf-8") as fh:
        for row in ds:
            msgs=normalize_messages(row.get("messages") or [])
            if len(msgs)<2 or not any(x["role"]=="assistant" for x in msgs):
                continue
            fh.write(json.dumps({"messages":msgs,"source":DEV_SOURCES["main_sft_tr_knowledge"]["repo"]},ensure_ascii=False)+"\n")
            added+=1
            if added>=limit:
                break
    if added < int(limit*0.90):
        raise RuntimeError(f"Turkish knowledge SFT underfilled {added}/{limit}")
    return {"rows":added,"source":DEV_SOURCES["main_sft_tr_knowledge"]["repo"]}


def append_tool_100k(path: Path, limit: int) -> dict:
    added=0
    ds=load_dataset(DEV_SOURCES["coder_tools_100k"]["repo"],split="train",streaming=True)
    with path.open("a",encoding="utf-8") as fh:
        for row in ds:
            msgs=normalize_messages(row.get("messages") or [])
            if not msgs or not any("<|tool_call|>" in x["content"] for x in msgs if x["role"]=="assistant"):
                continue
            tools=row.get("tools") or []
            if tools and (not msgs or msgs[0]["role"]!="system"):
                msgs.insert(0,{
                    "role":"system",
                    "content":"Available tools (JSON schema):\n"+compact_json(tools)
                    +"\nUse structured tool calls when a tool is required.",
                })
            fh.write(json.dumps({"messages":msgs,"source":DEV_SOURCES["coder_tools_100k"]["repo"]},ensure_ascii=False)+"\n")
            added+=1
            if added>=limit:
                break
    if added < int(limit*0.85):
        raise RuntimeError(f"tool 100k SFT underfilled {added}/{limit}")
    return {"rows":added,"source":DEV_SOURCES["coder_tools_100k"]["repo"]}


def count_jsonl(path: Path) -> int:
    with path.open("r",encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="prepared_v07_dev_text")
    ap.add_argument("--owner",required=True)
    ap.add_argument("--main-bytes",type=int,default=3_072_000_000)
    ap.add_argument("--coder-bytes",type=int,default=512_000_000)
    ap.add_argument("--vocab-size",type=int,default=16_384)
    ap.add_argument("--en-sft-rows",type=int,default=200_000)
    ap.add_argument("--tr-knowledge-rows",type=int,default=180_000)
    ap.add_argument("--coder-code-rows",type=int,default=150_000)
    ap.add_argument("--xlam-tool-rows",type=int,default=60_000)
    ap.add_argument("--tool100k-rows",type=int,default=100_000)
    args=ap.parse_args()

    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats={}
    print("V07_DEV_TEXT main_raw_start",flush=True)
    stats["main_raw"]=prepare_main_raw(out,args.main_bytes)
    print("V07_DEV_TEXT main_raw_done",json.dumps(stats["main_raw"]),flush=True)

    print("V07_DEV_TEXT coder_raw_start",flush=True)
    stats["coder_raw"]=prepare_coder_raw(out,args.coder_bytes)
    print("V07_DEV_TEXT coder_raw_done",json.dumps(stats["coder_raw"]),flush=True)

    stats["main_sft_base"]=prepare_main_sft(out,60_000,args.en_sft_rows)
    stats["main_sft_tr_knowledge"]=append_turkish_knowledge(out/"main_sft.jsonl",args.tr_knowledge_rows)
    stats["main_sft_raw"]={
        "file":"main_sft.jsonl",
        "rows":count_jsonl(out/"main_sft.jsonl"),
        "file_bytes":(out/"main_sft.jsonl").stat().st_size,
    }
    print("V07_DEV_TEXT main_sft_done",json.dumps(stats["main_sft_raw"]),flush=True)

    stats["coder_sft_base"]=prepare_coder_sft(out,args.coder_code_rows,args.xlam_tool_rows,6_000)
    stats["coder_sft_tool100k"]=append_tool_100k(out/"coder_sft.jsonl",args.tool100k_rows)
    stats["coder_sft_raw"]={
        "file":"coder_sft.jsonl",
        "rows":count_jsonl(out/"coder_sft.jsonl"),
        "file_bytes":(out/"coder_sft.jsonl").stat().st_size,
    }
    print("V07_DEV_TEXT coder_sft_done",json.dumps(stats["coder_sft_raw"]),flush=True)

    main_sample=out/"tokenizer_main_sample.txt"
    copy_prefix(out/"main.raw.txt",main_sample,384_000_000)
    stats["tokenizer"]=train_tokenizer(out,main_sample,out/"tokenizer_code_sample.txt",args.vocab_size)
    tok=Tokenizer.from_file(str(out/"tokenizer.json"))

    stats["main"]=encode_main(tok,out/"main.raw.txt",out/"main_train.u16")
    stats["coder"]=encode_coder_pretrain(tok,out/"coder_pretrain.jsonl",out/"coder_train.u16")
    stats["main_sft"]=encode_chat_sft(tok,out/"main_sft.jsonl",out/"main_sft_tokens.u16",out/"main_sft_mask.u8")
    stats["coder_sft"]=encode_chat_sft(tok,out/"coder_sft.jsonl",out/"coder_sft_tokens.u16",out/"coder_sft_mask.u8")

    for name in (
        "main.raw.txt","coder_pretrain.jsonl","tokenizer_main_sample.txt",
        "tokenizer_code_sample.txt","main_sft.jsonl","coder_sft.jsonl",
    ):
        (out/name).unlink(missing_ok=True)

    manifest={
        "pipeline_version":"v0.7-dev-text",
        "training_pipeline":"v0.7",
        "owner":args.owner,
        "sources":{**SOURCES,**DEV_SOURCES},
        "stats":stats,
        "format":{
            "main_train.u16":"uint16 BPE token IDs",
            "coder_train.u16":"uint16 BPE token IDs; code formatting preserved",
            "main_sft_*":"packed bilingual chat SFT + assistant mask",
            "coder_sft_*":"packed code/tool SFT + assistant mask",
        },
    }
    (out/"sources.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    meta={
        "title":"FLM v0.7 DEV Massive Text and Tool Data",
        "id":f"{args.owner}/flm-v07-dev-text",
        "licenses":[{"name":"other"}],
    }
    (out/"dataset-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("V07_DEV_TEXT_COMPLETE",json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
