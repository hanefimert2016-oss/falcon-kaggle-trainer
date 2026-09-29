#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPECIAL=[
    "<pad>","<unk>","<bos>","<eos>","<doc>",
    "<|system|>","<|user|>","<|assistant|>",
    "<|tool_call|>","<|tool_result|>","<|tool_end|>",
    "<|semantic_ir|>","<|semantic_end|>",
    "<|core_result|>","<|core_end|>",
    "<|plan|>","<|plan_end|>","<|final|>","<|end|>",
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--dataset",default="flm-v07-zero-train-data-r1")
    ap.add_argument("--out",default="kernel_v07_tokenizer_only")
    ap.add_argument("--vocab-size",type=int,default=32768)
    args=ap.parse_args()

    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    script=f'''# Auto-generated FLM tokenizer-only Kaggle job.
from pathlib import Path
import json
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL={SPECIAL!r}
VOCAB={int(args.vocab_size)}
root=Path("/kaggle/input")
candidates=[]
for name in ("main.raw.txt","coder_pretrain.jsonl","main_sft.jsonl","coder_sft.jsonl"):
    candidates.extend(root.rglob(name))
paths=[]
seen=set()
for p in candidates:
    key=str(p.resolve())
    if key not in seen and p.is_file():
        seen.add(key); paths.append(str(p))
if not paths:
    raise SystemExit("FLM_TOKENIZER_NO_CORPUS")
tok=Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
tok.decoder=ByteLevelDecoder()
trainer=BpeTrainer(vocab_size=VOCAB,min_frequency=2,special_tokens=SPECIAL)
tok.train(paths,trainer)
out=Path("/kaggle/working/flm-tokenizer")
out.mkdir(parents=True,exist_ok=True)
tok.save(str(out/"tokenizer.json"))
meta={{
    "format":"flm-tokenizer-v1",
    "training_kind":"tokenizer-only-no-neural-model",
    "vocab_size":tok.get_vocab_size(),
    "special_ids":{{x:tok.token_to_id(x) for x in SPECIAL}},
    "corpus_files":[Path(x).name for x in paths],
}}
(out/"tokenizer_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
print("FLM_TOKENIZER_ONLY_RESULT="+json.dumps(meta,ensure_ascii=False))
'''
    (out/"train_tokenizer.py").write_text(script,encoding="utf-8")
    meta={
        "id":f"{args.owner}/falcon-flm-v07-tokenizer-only",
        "title":"Falcon FLM v07 Tokenizer Only",
        "code_file":"train_tokenizer.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":False,
        "enable_internet":False,
        "dataset_sources":[f"{args.owner}/{args.dataset}"],
        "kernel_sources":[],
        "competition_sources":[],
        "model_sources":[],
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "kernel":meta["id"],
        "training_kind":"tokenizer-only",
        "neural_model_training":False,
        "vocab_size":args.vocab_size,
        "dataset_sources":meta["dataset_sources"],
    }))


if __name__=="__main__":
    main()
