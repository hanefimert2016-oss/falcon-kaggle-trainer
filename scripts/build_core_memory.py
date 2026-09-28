#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from flm.models.core_memory import CoreMemoryBuilder


def iter_u16(path: Path, chunk_tokens: int):
    arr=np.memmap(path,mode="r",dtype="<u2")
    for start in range(0,len(arr),chunk_tokens):
        yield np.asarray(arr[start:start+chunk_tokens],dtype=np.int64)


def iter_text(path: Path, tok: Tokenizer):
    with path.open("r",encoding="utf-8",errors="ignore") as fh:
        for line in fh:
            if line.strip():
                yield np.asarray(
                    tok.encode(line,add_special_tokens=False).ids,
                    dtype=np.int64,
                )


def main():
    ap=argparse.ArgumentParser(
        description="Compile raw/tokenized data into FLM's internal non-RAG CoreMemory."
    )
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens-u16",type=Path)
    src.add_argument("--text",type=Path)
    ap.add_argument("--tokenizer",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--vocab-size",type=int)
    ap.add_argument("--order",type=int,default=4)
    ap.add_argument("--slots",type=int,default=1<<20)
    ap.add_argument("--top-k",type=int,default=4)
    ap.add_argument("--max-tokens",type=int,default=0)
    ap.add_argument("--chunk-tokens",type=int,default=1_000_000)
    args=ap.parse_args()

    tok=None
    if args.text:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required with --text")
        tok=Tokenizer.from_file(str(args.tokenizer))
        vocab=tok.get_vocab_size()
        chunks=iter_text(args.text,tok)
    else:
        if not args.vocab_size:
            raise SystemExit("--vocab-size is required with --tokens-u16")
        vocab=args.vocab_size
        chunks=iter_u16(args.tokens_u16,args.chunk_tokens)

    builder=CoreMemoryBuilder(
        vocab_size=vocab,order=args.order,slots=args.slots,top_k=args.top_k
    )
    remaining=args.max_tokens if args.max_tokens>0 else None
    for chunk in chunks:
        if remaining is not None:
            if remaining<=0:
                break
            chunk=chunk[:remaining]
            remaining-=len(chunk)
        builder.ingest(chunk.tolist())

    bank,meta=builder.build()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    bank.save(args.out,meta=meta.__dict__)
    print("FLM_CORE_MEMORY_READY="+json.dumps({
        **meta.__dict__,
        "path":str(args.out),
        "bytes":args.out.stat().st_size,
    },ensure_ascii=False))


if __name__=="__main__":
    main()
