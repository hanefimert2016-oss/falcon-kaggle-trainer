#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from datasets import load_dataset
from tokenizers import Tokenizer

from flm.data.semantic_curriculum import (
    SEMANTIC_MODES,
    append_semantic_curriculum,
    copy_similarity,
    messages_for,
)
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
    "main_wiki_tr": {
        "repo": "wikimedia/wikipedia",
        "config": "20231101.tr",
        "license": "cc-by-sa-3.0/gfdl",
        "role": "clean Turkish encyclopedic pretraining",
    },
    "main_wiki_en": {
        "repo": "wikimedia/wikipedia",
        "config": "20231101.en",
        "license": "cc-by-sa-3.0/gfdl",
        "role": "clean English encyclopedic pretraining",
    },
    "coder_agent_menv": {
        "repo": "AmanPriyanshu/tool-reasoning-sft-CODING-MEnvData-SWE-Trajectory-data-cleaned-rectified",
        "license": "apache-2.0",
        "role": "validated multi-step software-engineering planning and tool trajectories",
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


def append_wikipedia_knowledge(path: Path, source_key: str, target_bytes: int) -> dict:
    source=DEV_SOURCES[source_key]
    written=rows=0
    ds=load_dataset(source["repo"],source["config"],split="train",streaming=True)
    with path.open("a",encoding="utf-8") as fh:
        for row in ds:
            text=" ".join(clean_text(row.get("text") or "").split())
            title=" ".join(clean_text(row.get("title") or "").split())
            if len(text)<200:
                continue
            doc=(title+"\n"+text).strip()
            raw=(doc+"\n").encode("utf-8","ignore")
            remain=target_bytes-written
            if remain<=0:
                break
            if len(raw)>remain:
                raw=raw[:remain].decode("utf-8","ignore").encode("utf-8")
            if not raw:
                continue
            fh.write(raw.decode("utf-8","ignore"))
            written+=len(raw)
            rows+=1
            if written>=target_bytes-4:
                break
    if written < int(target_bytes*0.90):
        raise RuntimeError(f"{source_key} underfilled {written}/{target_bytes}")
    return {"bytes":written,"articles":rows,"source":source["repo"],"config":source["config"]}


def _strip_wrapped(text: str, start: str, end: str) -> str:
    text=clean_text(text)
    if text.startswith(start) and text.endswith(end):
        text=text[len(start):-len(end)]
    return text.strip()


def normalize_agentic_coder_messages(raw):
    if isinstance(raw,str):
        try:
            raw=json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw,list):
        return []
    out=[]
    for m in raw:
        if not isinstance(m,dict):
            continue
        role=str(m.get("role") or "").strip().lower()
        content=clean_text(m.get("content") or "")
        if not content:
            continue
        if len(content)>16000:
            content=content[:16000]
        if role in {"system","user"}:
            out.append({"role":role,"content":content})
        elif role=="reasoning":
            content=_strip_wrapped(content,"<think>","</think>")
            out.append({
                "role":"assistant",
                "content":"<|plan|>\n"+content+"\n<|plan_end|>",
            })
        elif role=="tool_call":
            content=content.replace("<tool_call>","<|tool_call|>").replace("</tool_call>","<|tool_end|>")
            if "<|tool_call|>" not in content:
                content="<|tool_call|>"+content+"<|tool_end|>"
            out.append({"role":"assistant","content":content})
        elif role=="tool_output":
            content=content.replace("<tool_response>","").replace("</tool_response>","").strip()
            out.append({
                "role":"tool",
                "content":"<|tool_result|>"+content+"<|tool_end|>",
            })
        elif role=="answer":
            content=_strip_wrapped(content,"<answer>","</answer>")
            out.append({"role":"assistant","content":"<|final|>\n"+content})
        elif role=="assistant":
            out.append({"role":"assistant","content":content})
        elif role=="tool":
            out.append({"role":"tool","content":content})
    return out


def compact_agentic_for_context(messages, max_chars: int = 12_000):
    """Keep a coherent multi-step prefix plus the real final answer within ~4K tokens."""
    if not messages:
        return []
    prefix=[]
    rest=[]
    final=None
    for m in messages:
        if m["role"] in {"system","user"} and len(prefix)<2:
            x=dict(m); x["content"]=x["content"][:2000]
            prefix.append(x)
        elif m["role"]=="assistant" and "<|final|>" in m["content"]:
            final=dict(m); final["content"]=final["content"][:2200]
        else:
            rest.append(dict(m))

    used=sum(len(m["content"]) for m in prefix)
    reserve=(len(final["content"]) if final else 0)+256
    body=[]
    for m in rest:
        room=max_chars-used-reserve
        if room<=256:
            break
        content=m["content"][:min(len(m["content"]),2400,room)]
        if not content:
            continue
        x=dict(m); x["content"]=content
        body.append(x)
        used+=len(content)
    out=prefix+body
    if final is not None and final not in out:
        out.append(final)
    return out


def append_coder_agent_trajectories(path: Path, limit: int) -> dict:
    source=DEV_SOURCES["coder_agent_menv"]
    ds=load_dataset(source["repo"],split="train",streaming=True)
    added=0
    plan_turns=tool_turns=final_turns=0
    with path.open("a",encoding="utf-8") as fh:
        for row in ds:
            msgs=compact_agentic_for_context(
                normalize_agentic_coder_messages(row.get("messages"))
            )
            if len(msgs)<8:
                continue
            plans=sum("<|plan|>" in m["content"] for m in msgs if m["role"]=="assistant")
            tools=sum("<|tool_call|>" in m["content"] for m in msgs if m["role"]=="assistant")
            finals=sum("<|final|>" in m["content"] for m in msgs if m["role"]=="assistant")
            if plans<2 or tools<1 or finals<1:
                continue
            fh.write(json.dumps({"messages":msgs,"source":source["repo"]},ensure_ascii=False)+"\n")
            added+=1
            plan_turns+=plans
            tool_turns+=tools
            final_turns+=finals
            if added>=limit:
                break
    if added < int(limit*0.85):
        raise RuntimeError(f"agentic coder trajectories underfilled {added}/{limit}")
    return {
        "rows":added,
        "plan_turns":plan_turns,
        "tool_turns":tool_turns,
        "final_turns":final_turns,
        "source":source["repo"],
    }


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
    ap.add_argument("--wiki-tr-bytes",type=int,default=768_000_000)
    ap.add_argument("--wiki-en-bytes",type=int,default=768_000_000)
    ap.add_argument("--coder-bytes",type=int,default=768_000_000)
    ap.add_argument("--vocab-size",type=int,default=32_768)
    ap.add_argument("--en-sft-rows",type=int,default=300_000)
    ap.add_argument("--tr-knowledge-rows",type=int,default=320_000)
    ap.add_argument("--coder-code-rows",type=int,default=250_000)
    ap.add_argument("--xlam-tool-rows",type=int,default=100_000)
    ap.add_argument("--tool100k-rows",type=int,default=100_000)
    ap.add_argument("--coder-agent-rows",type=int,default=6_000)
    ap.add_argument("--semantic-sft-rows",type=int,default=800_000)
    args=ap.parse_args()

    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats={}

    # Fail fast before downloading/processing gigabytes if the executable
    # Semantic IR/Core curriculum is inconsistent.
    semantic_preflight=[]
    for sample in range(SEMANTIC_MODES*2):
        mode=sample%SEMANTIC_MODES
        language="tr" if sample<SEMANTIC_MODES else "en"
        messages=messages_for(sample)
        user=next(m["content"] for m in messages if m.get("role")=="user")
        final=next(
            m["content"].split("<|final|>",1)[-1].strip()
            for m in messages
            if m.get("role")=="assistant" and "<|final|>" in m.get("content","")
        )
        semantic_preflight.append({
            "mode":mode,
            "language":language,
            "messages":len(messages),
            "has_ir":any("<|semantic_ir|>" in m.get("content","") for m in messages),
            "has_core":any("<|core_result|>" in m.get("content","") for m in messages),
            "copy_similarity":copy_similarity(user,final),
        })
    if not all(
        x["messages"]>=5 and x["has_ir"] and x["has_core"]
        and x["copy_similarity"]<0.88
        for x in semantic_preflight
    ):
        raise RuntimeError(f"semantic curriculum preflight failed: {semantic_preflight}")
    stats["semantic_preflight"]={
        "modes":SEMANTIC_MODES,
        "languages":["tr","en"],
        "ok":True,
        "anti_copy_max":max(x["copy_similarity"] for x in semantic_preflight),
    }
    print("V07_DEV_TEXT semantic_preflight_ok",flush=True)

    print("V07_DEV_TEXT main_raw_start",flush=True)
    stats["main_raw"]=prepare_main_raw(out,args.main_bytes)
    stats["main_wiki_tr"]=append_wikipedia_knowledge(out/"main.raw.txt","main_wiki_tr",args.wiki_tr_bytes)
    stats["main_wiki_en"]=append_wikipedia_knowledge(out/"main.raw.txt","main_wiki_en",args.wiki_en_bytes)
    stats["main_raw"]["bytes_total_with_knowledge"]=(out/"main.raw.txt").stat().st_size
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

    # Keep response-synthesis/IR training independent from ordinary instruction
    # data so the single InterfaceTransformer can sample it at an explicit ratio.
    semantic_raw=out/"semantic_sft.jsonl"
    stats["semantic_interface_sft"]=append_semantic_curriculum(
        semantic_raw,args.semantic_sft_rows
    )
    stats["semantic_sft_raw"]={
        "file":"semantic_sft.jsonl",
        "rows":count_jsonl(semantic_raw),
        "file_bytes":semantic_raw.stat().st_size,
    }
    print("V07_DEV_TEXT semantic_sft_done",json.dumps(stats["semantic_sft_raw"]),flush=True)

    stats["coder_sft_base"]=prepare_coder_sft(out,args.coder_code_rows,args.xlam_tool_rows,6_000)
    stats["coder_sft_tool100k"]=append_tool_100k(out/"coder_sft.jsonl",args.tool100k_rows)
    stats["coder_sft_agentic"]=append_coder_agent_trajectories(out/"coder_sft.jsonl",args.coder_agent_rows)
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

    # Encode and delete large raw intermediates stage-by-stage so the GitHub
    # runner does not need space for every raw + tokenized corpus at once.
    (out/"tokenizer_main_sample.txt").unlink(missing_ok=True)
    (out/"tokenizer_code_sample.txt").unlink(missing_ok=True)

    stats["main"]=encode_main(tok,out/"main.raw.txt",out/"main_train.u16")
    (out/"main.raw.txt").unlink(missing_ok=True)

    stats["coder"]=encode_coder_pretrain(tok,out/"coder_pretrain.jsonl",out/"coder_train.u16")
    (out/"coder_pretrain.jsonl").unlink(missing_ok=True)

    stats["main_sft"]=encode_chat_sft(tok,out/"main_sft.jsonl",out/"main_sft_tokens.u16",out/"main_sft_mask.u8")
    (out/"main_sft.jsonl").unlink(missing_ok=True)

    stats["semantic_sft"]=encode_chat_sft(tok,out/"semantic_sft.jsonl",out/"semantic_sft_tokens.u16",out/"semantic_sft_mask.u8")
    (out/"semantic_sft.jsonl").unlink(missing_ok=True)

    stats["coder_sft"]=encode_chat_sft(tok,out/"coder_sft.jsonl",out/"coder_sft_tokens.u16",out/"coder_sft_mask.u8")
    (out/"coder_sft.jsonl").unlink(missing_ok=True)

    manifest={
        "pipeline_version":"v0.7-dev-text",
        "data_revision":6,
        "training_pipeline":"v0.7",
        "owner":args.owner,
        "sources":{**SOURCES,**DEV_SOURCES},
        "stats":stats,
        "format":{
            "main_train.u16":"uint16 BPE token IDs",
            "coder_train.u16":"uint16 BPE token IDs; code formatting preserved",
            "main_sft_*":"packed bilingual general instruction SFT + assistant mask",
            "semantic_sft_*":"packed Semantic IR/Core response-synthesis SFT + assistant mask",
            "coder_sft_*":"packed code/tool/agentic-plan SFT + assistant mask",
            "semantic_tokens":["<|semantic_ir|>","<|semantic_end|>","<|core_result|>","<|core_end|>"],
            "planning_tokens":["<|plan|>","<|plan_end|>","<|final|>"],
        },
    }
    (out/"sources.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    meta={
        "title":"FLM v0.7 Semantic Interface Text r6",
        "id":f"{args.owner}/flm-v07-semantic-text-r6",
        "licenses":[{"name":"other"}],
    }
    (out/"dataset-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("V07_DEV_TEXT_COMPLETE",json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
