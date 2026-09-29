#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from flm.data.prepare_v07 import prepare_main_raw, prepare_coder_raw


SOURCES={
    "wikidata":{
        "repo":"RJZ/wikidata_triple_en",
        "license":"cc0-1.0",
        "role":"labeled world-knowledge triples for direct Semantic Memory ingestion",
    },
    "turkish_thesaurus":{
        "repo":"agmmnn/turkish-thesaurus-synonyms-antonyms",
        "license":"cc-by-sa-4.0",
        "role":"Turkish synonym/antonym lexicon for grammar and surface variation",
    },
}


def build_wikidata(path:Path,max_rows:int)->dict:
    rows=0
    entities=set()
    predicates=set()
    ds=load_dataset(SOURCES["wikidata"]["repo"],split="train",streaming=True)
    with path.open("w",encoding="utf-8") as fh:
        for raw in ds:
            obj={
                "s_id":str(raw.get("s_id") or "").strip(),
                "r_id":str(raw.get("r_id") or "").strip(),
                "e_id":str(raw.get("e_id") or "").strip(),
                "s_label":str(raw.get("s_label") or "").strip(),
                "r_label":str(raw.get("r_label") or "").strip(),
                "e_label":str(raw.get("e_label") or "").strip(),
            }
            if not all(obj[k] for k in ("s_id","r_id","e_id","s_label","r_label","e_label")):
                continue
            fh.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n")
            rows+=1
            entities.add(obj["s_id"]); entities.add(obj["e_id"])
            predicates.add(obj["r_id"])
            if rows>=max_rows:
                break
    if rows<min(max_rows,100_000):
        raise RuntimeError(f"wikidata underfilled {rows}/{max_rows}")
    return {
        "file":path.name,"rows":rows,
        "unique_entities":len(entities),"unique_predicates":len(predicates),
        **SOURCES["wikidata"],
    }


def build_turkish_lexicon(path:Path)->dict:
    rows=0
    synonym_edges=0
    antonym_edges=0
    ds=load_dataset(SOURCES["turkish_thesaurus"]["repo"],split="train")
    with path.open("w",encoding="utf-8") as fh:
        for raw in ds:
            word=str(raw.get("word") or "").strip()
            synonyms=[str(x).strip() for x in (raw.get("synonyms") or []) if str(x).strip()]
            antonyms=[str(x).strip() for x in (raw.get("antonyms") or []) if str(x).strip()]
            if not word or (not synonyms and not antonyms):
                continue
            obj={"word":word,"synonyms":synonyms,"antonyms":antonyms}
            fh.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n")
            rows+=1
            synonym_edges+=len(synonyms)
            antonym_edges+=len(antonyms)
    if rows<20_000:
        raise RuntimeError(f"Turkish lexicon unexpectedly small: {rows}")
    return {
        "file":path.name,"rows":rows,
        "synonym_edges":synonym_edges,"antonym_edges":antonym_edges,
        **SOURCES["turkish_thesaurus"],
    }


def main()->int:
    ap=argparse.ArgumentParser(description="Build FLM strict zero-train data. No tokenizer/model training here.")
    ap.add_argument("--out",default="prepared_zero_train")
    ap.add_argument("--owner",required=True)
    ap.add_argument("--main-bytes",type=int,default=3_072_000_000)
    ap.add_argument("--coder-bytes",type=int,default=768_000_000)
    ap.add_argument("--wikidata-rows",type=int,default=1_000_000)
    args=ap.parse_args()

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)

    stats={}
    print("ZERO_TRAIN main_raw_start",flush=True)
    stats["tokenizer_main_corpus"]=prepare_main_raw(out,args.main_bytes)
    print("ZERO_TRAIN main_raw_done",json.dumps(stats["tokenizer_main_corpus"]),flush=True)

    print("ZERO_TRAIN coder_raw_start",flush=True)
    stats["tokenizer_code_corpus"]=prepare_coder_raw(out,args.coder_bytes)
    print("ZERO_TRAIN coder_raw_done",json.dumps(stats["tokenizer_code_corpus"]),flush=True)

    print("ZERO_TRAIN wikidata_start",flush=True)
    stats["wikidata"]=build_wikidata(out/"semantic_wikidata.jsonl",args.wikidata_rows)
    print("ZERO_TRAIN wikidata_done",json.dumps(stats["wikidata"]),flush=True)

    print("ZERO_TRAIN turkish_lexicon_start",flush=True)
    stats["turkish_lexicon"]=build_turkish_lexicon(out/"turkish_lexicon.jsonl")
    print("ZERO_TRAIN turkish_lexicon_done",json.dumps(stats["turkish_lexicon"]),flush=True)

    # The old coder helper emits this tokenizer sample. Kaggle can use the full
    # code corpus directly, so remove the redundant sample from the dataset.
    (out/"tokenizer_code_sample.txt").unlink(missing_ok=True)

    manifest={
        "format":"flm-zero-train-data-v1",
        "data_revision":1,
        "owner":args.owner,
        "architecture":"tokenizer + SemanticCompiler + SemanticMemory + ReasoningCore + ResponseComposer",
        "training_steps":0,
        "model_training":False,
        "tokenizer_trained_here":False,
        "raw_prose_answer_memory":False,
        "source_hidden_semantic_memory":True,
        "stats":stats,
    }
    (out/"zero_train_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    (out/"dataset-metadata.json").write_text(json.dumps({
        "title":"FLM v0.7 Zero Train Data r1",
        "id":f"{args.owner}/flm-v07-zero-train-data-r1",
        "licenses":[{"name":"other"}],
    },indent=2)+"\n",encoding="utf-8")
    print("FLM_ZERO_TRAIN_DATA_READY="+json.dumps(manifest,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
