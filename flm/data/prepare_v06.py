#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SOURCES = {
    "main_en": {
        "repo": "codelion/fineweb-edu-100M",
        "license": "odc-by",
        "role": "English educational web pretraining",
    },
    "main_tr": {
        "repo": "moganai/turkishfineweb2-cleaned",
        "license": "odc-by",
        "role": "cleaned/deduplicated Turkish FineWeb-2 pretraining",
    },
    "main_tr_fallback": {
        "repo": "HuggingFaceFW/fineweb-2",
        "config": "tur_Latn",
        "license": "odc-by",
        "role": "Turkish FineWeb-2 fallback",
    },
    "sft_tr_alpaca": {
        "repo": "TFLai/Turkish-Alpaca",
        "license": "apache-2.0",
        "role": "Turkish instruction-response SFT",
    },
    "sft_tr_merve": {
        "repo": "merve/turkish_instructions",
        "license": "apache-2.0",
        "role": "Turkish instruction-response SFT",
    },
    "coder_python": {
        "repo": "Nan-Do/code-search-net-python",
        "license": "apache-2.0",
        "role": "Python code/docstrings",
    },
    "coder_unreal": {
        "repo": "AdamCodd/unreal-engine-5-code",
        "license": "apache-2.0",
        "role": "Unreal Engine code/API",
    },
}

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>", "<|user|>", "<|assistant|>", "<|end|>"]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def append_doc(handle, text: str) -> int:
    text = one_line(text)
    if not text:
        return 0
    payload = (text + "\n").encode("utf-8", "ignore")
    handle.write(payload)
    return len(payload)


def stream_text(repo: str, *, config: str | None = None):
    if config:
        return load_dataset(repo, config, split="train", streaming=True)
    return load_dataset(repo, split="train", streaming=True)


def fill_text_source(path: Path, source: dict, target_bytes: int, seen: set[str]) -> dict:
    total = docs = skipped_dup = 0
    ds = stream_text(source["repo"], config=source.get("config"))
    with path.open("ab") as f:
        for row in ds:
            text = one_line(row.get("text") or row.get("content") or "")
            if len(text) < 180:
                continue
            sig = hashlib.blake2b(text[:4096].encode("utf-8", "ignore"), digest_size=12).hexdigest()
            if sig in seen:
                skipped_dup += 1
                continue
            seen.add(sig)
            payload = (text + "\n").encode("utf-8", "ignore")
            remain = target_bytes - total
            if remain <= 0:
                break
            if len(payload) > remain:
                payload = payload[:remain]
            f.write(payload)
            total += len(payload)
            docs += 1
            if total >= target_bytes:
                break
    return {"bytes": total, "documents": docs, "duplicates_skipped": skipped_dup}


def prepare_main(out: Path, target_bytes: int) -> dict:
    raw = out / "main.raw.txt"
    raw.unlink(missing_ok=True)
    # Equal language budget so Turkish is not drowned out by English.
    tr_budget = target_bytes // 2
    en_budget = target_bytes - tr_budget
    seen: set[str] = set()

    tr = None
    try:
        tr = fill_text_source(raw, SOURCES["main_tr"], tr_budget, seen)
    except Exception as exc:
        print(f"v06 Turkish primary source failed: {type(exc).__name__}: {exc}", flush=True)
        tr = fill_text_source(raw, SOURCES["main_tr_fallback"], tr_budget, seen)
    if tr["bytes"] < int(tr_budget * 0.95):
        raise RuntimeError(f"Turkish corpus underfilled: {tr}")

    en = fill_text_source(raw, SOURCES["main_en"], en_budget, seen)
    if en["bytes"] < int(en_budget * 0.95):
        raise RuntimeError(f"English corpus underfilled: {en}")

    return {
        "raw_file": raw.name,
        "bytes": raw.stat().st_size,
        "turkish": tr,
        "english": en,
        "mix": "50% Turkish / 50% English by byte budget",
    }


def coder_payload(instruction: str, code: str, source: str) -> str:
    instruction = one_line(instruction) or "Explain, complete, or improve this code."
    code = clean_text(code)
    return f"<source>{source}</source>\n<instruction>{instruction}</instruction>\n<code>\n{code}\n</code>"


def prepare_coder(out: Path, target_bytes: int) -> dict:
    raw = out / "coder.raw.txt"
    raw.unlink(missing_ok=True)
    total = 0
    counts = {"unreal": 0, "python": 0}
    with raw.open("ab") as f:
        unreal_budget = min(target_bytes // 4, 16_000_000)
        try:
            ds = stream_text(SOURCES["coder_unreal"]["repo"])
            for row in ds:
                code = str(row.get("code") or "").strip()
                if len(code) < 60:
                    continue
                desc = row.get("description") or row.get("className") or ""
                text = coder_payload(desc, code, "unreal-engine-5")
                payload = (one_line(text) + "\n").encode("utf-8", "ignore")
                if total + len(payload) > unreal_budget:
                    break
                f.write(payload)
                total += len(payload)
                counts["unreal"] += 1
        except Exception as exc:
            print(f"v06 optional Unreal source failed: {type(exc).__name__}: {exc}", flush=True)

        ds = stream_text(SOURCES["coder_python"]["repo"])
        for row in ds:
            code = str(row.get("code") or row.get("whole_func_string") or row.get("func_code_string") or "").strip()
            if len(code) < 80:
                continue
            desc = row.get("summary") or row.get("docstring") or row.get("func_documentation_string") or ""
            text = coder_payload(desc, code, "codesearchnet-python")
            payload = (one_line(text) + "\n").encode("utf-8", "ignore")
            remain = target_bytes - total
            if remain <= 0:
                break
            if len(payload) > remain:
                payload = payload[:remain]
            f.write(payload)
            total += len(payload)
            counts["python"] += 1
            if total >= target_bytes:
                break
    if total < int(target_bytes * 0.95):
        raise RuntimeError(f"coder corpus underfilled: {total}/{target_bytes}")
    return {"raw_file": raw.name, "bytes": total, "records": counts}


def normalize_sft_row(row: dict) -> tuple[str, str] | None:
    # Some CSV-backed HF datasets preserve whitespace around column names.
    row = {str(k).strip(): v for k, v in row.items()}
    instruction = row.get("instruction") or row.get("talimat") or row.get("question") or ""
    extra = row.get("input") or row.get("giriş") or row.get("giris") or ""
    output = row.get("output") or row.get("çıktı") or row.get("cikti") or row.get("answer") or ""
    instruction = clean_text(instruction)
    extra = clean_text(extra)
    output = clean_text(output)
    if not instruction or not output:
        return None
    user = instruction if not extra else f"{instruction}\n\nBağlam:\n{extra}"
    return user, output


def prepare_sft(out: Path, max_rows_per_source: int = 60_000) -> dict:
    path = out / "main_sft.jsonl"
    path.unlink(missing_ok=True)
    counts = {}
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as f:
        for key in ("sft_tr_alpaca", "sft_tr_merve"):
            source = SOURCES[key]
            added = dup = 0
            ds = stream_text(source["repo"])
            for row in ds:
                pair = normalize_sft_row(row)
                if pair is None:
                    continue
                user, assistant = pair
                sig = hashlib.blake2b((user + "\n" + assistant).encode("utf-8", "ignore"), digest_size=12).hexdigest()
                if sig in seen:
                    dup += 1
                    continue
                seen.add(sig)
                f.write(json.dumps({"user": user, "assistant": assistant, "source": source["repo"]}, ensure_ascii=False) + "\n")
                added += 1
                if added >= max_rows_per_source:
                    break
            counts[key] = {"rows": added, "duplicates_skipped": dup}
            print(f"v06_data sft_source={key} rows={added} dup={dup}", flush=True)
    rows = sum(x["rows"] for x in counts.values())
    if rows < 80_000:
        raise RuntimeError(f"SFT corpus unexpectedly small: {rows}")
    return {"file": path.name, "rows": rows, "sources": counts, "bytes": path.stat().st_size}


def train_tokenizer(out: Path, main_raw: Path, coder_raw: Path, vocab_size: int) -> dict:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tok.train([str(main_raw), str(coder_raw)], trainer)
    path = out / "tokenizer.json"
    tok.save(str(path))
    actual = tok.get_vocab_size()
    if actual < min(vocab_size, 4096):
        raise RuntimeError(f"tokenizer vocab too small: {actual}")
    meta = {name: tok.token_to_id(name) for name in SPECIAL}
    meta["vocab_size"] = actual
    (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"file": path.name, "vocab_size": actual, "special_ids": meta}


def encode_lines(tok: Tokenizer, src: Path, dst: Path) -> dict:
    eos = tok.token_to_id("<eos>")
    if eos is None:
        raise RuntimeError("missing eos token")
    total = docs = 0
    with src.open("r", encoding="utf-8", errors="ignore") as f, dst.open("wb") as out:
        for line in f:
            text = line.strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False).ids + [eos]
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(out)
            total += len(ids)
            docs += 1
    return {"file": dst.name, "tokens": total, "documents": docs, "bytes": dst.stat().st_size}


def encode_sft(tok: Tokenizer, src: Path, out: Path) -> dict:
    tokens_path = out / "main_sft_tokens.u16"
    mask_path = out / "main_sft_mask.u8"
    ids = {name: tok.token_to_id(name) for name in SPECIAL}
    required = ("<bos>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>")
    if any(ids[x] is None for x in required):
        raise RuntimeError(f"missing special IDs: {ids}")

    total = supervised = rows = 0
    with src.open("r", encoding="utf-8") as f, tokens_path.open("wb") as tf, mask_path.open("wb") as mf:
        for line in f:
            row = json.loads(line)
            user_ids = tok.encode(row["user"], add_special_tokens=False).ids
            answer_ids = tok.encode(row["assistant"], add_special_tokens=False).ids
            prompt = [ids["<bos>"], ids["<|user|>"]] + user_ids + [ids["<|assistant|>"]]
            answer = answer_ids + [ids["<|end|>"], ids["<eos>"]]
            seq = prompt + answer
            mask = [0] * len(prompt) + [1] * len(answer)
            np.asarray(seq, dtype=np.uint16).tofile(tf)
            np.asarray(mask, dtype=np.uint8).tofile(mf)
            total += len(seq)
            supervised += len(answer)
            rows += 1
    return {
        "tokens_file": tokens_path.name,
        "mask_file": mask_path.name,
        "tokens": total,
        "supervised_tokens": supervised,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prepared_v06")
    ap.add_argument("--owner", required=True)
    ap.add_argument("--main-bytes", type=int, default=128_000_000)
    ap.add_argument("--coder-bytes", type=int, default=64_000_000)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--sft-max-rows-per-source", type=int, default=60_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*"):
        if p.is_file():
            p.unlink()

    stats = {}
    print("v06_data main_start", flush=True)
    stats["main_raw"] = prepare_main(out, args.main_bytes)
    print("v06_data main_done", json.dumps(stats["main_raw"]), flush=True)

    print("v06_data coder_start", flush=True)
    stats["coder_raw"] = prepare_coder(out, args.coder_bytes)
    print("v06_data coder_done", json.dumps(stats["coder_raw"]), flush=True)

    print("v06_data sft_start", flush=True)
    stats["sft_raw"] = prepare_sft(out, args.sft_max_rows_per_source)
    print("v06_data sft_done", json.dumps(stats["sft_raw"]), flush=True)

    print("v06_data tokenizer_start", flush=True)
    stats["tokenizer"] = train_tokenizer(out, out/"main.raw.txt", out/"coder.raw.txt", args.vocab_size)
    tok = Tokenizer.from_file(str(out/"tokenizer.json"))
    print("v06_data tokenizer_done", json.dumps(stats["tokenizer"]), flush=True)

    stats["main"] = encode_lines(tok, out/"main.raw.txt", out/"main_train.u16")
    stats["coder"] = encode_lines(tok, out/"coder.raw.txt", out/"coder_train.u16")
    stats["sft"] = encode_sft(tok, out/"main_sft.jsonl", out)

    # Raw corpora are only tokenizer/preprocessing intermediates; encoded tokens
    # plus source metadata are sufficient for reproducible GPU training.
    (out/"main.raw.txt").unlink(missing_ok=True)
    (out/"coder.raw.txt").unlink(missing_ok=True)

    manifest = {
        "pipeline_version": "v0.6",
        "owner": args.owner,
        "sources": SOURCES,
        "stats": stats,
        "format": {
            "main_train.u16": "little-endian uint16 token IDs",
            "coder_train.u16": "little-endian uint16 token IDs",
            "main_sft_tokens.u16": "packed SFT token IDs",
            "main_sft_mask.u8": "1 means assistant/supervised token",
        },
    }
    (out/"sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "title": "Falcon FLM v06 multilingual BPE text data",
        "id": f"{args.owner}/flm-hf-v06",
        "licenses": [{"name": "other"}],
    }
    (out/"dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("v06_data_complete", json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
