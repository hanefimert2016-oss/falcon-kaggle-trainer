#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
from datasets import load_dataset
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SOURCES = {
    "main_tr": {
        "repo": "moganai/turkishfineweb2-cleaned",
        "license": "odc-by",
        "role": "Turkish cleaned web pretraining",
    },
    "main_tr_fallback": {
        "repo": "HuggingFaceFW/fineweb-2",
        "config": "tur_Latn",
        "license": "odc-by",
        "role": "Turkish FineWeb-2 fallback",
    },
    "main_en": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "license": "odc-by",
        "role": "large English educational pretraining",
    },
    "main_sft_tr_alpaca": {
        "repo": "TFLai/Turkish-Alpaca",
        "license": "apache-2.0",
        "role": "Turkish instruction-response SFT",
    },
    "main_sft_tr_merve": {
        "repo": "merve/turkish_instructions",
        "license": "apache-2.0",
        "role": "Turkish instruction-response SFT",
    },
    "main_sft_en_ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "license": "mit",
        "role": "English multi-turn chat SFT",
    },
    "coder_python": {
        "repo": "Nan-Do/code-search-net-python",
        "license": "apache-2.0",
        "role": "Python code with docstrings; formatting preserved",
    },
    "coder_unreal": {
        "repo": "AdamCodd/unreal-engine-5-code",
        "license": "apache-2.0",
        "role": "Unreal Engine code/API; formatting preserved",
    },
    "coder_tools": {
        "repo": "product-science/xlam-function-calling-60k-raw",
        "license": "apache-2.0",
        "role": "structured function/tool-call SFT",
    },
    "computer_rexx": {
        "repo": "REXX-NEW/computer-use",
        "license": "apache-2.0",
        "role": "successful OSWorld desktop trajectories with screenshots and pyautogui actions",
    },
}

SPECIAL = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<doc>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|tool_result|>", "<|tool_end|>",
    "<|plan|>", "<|plan_end|>", "<|final|>", "<|end|>",
]


def clean_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\x00", " ")
    text = re.sub(r"[\r\f\v]+", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def one_line(value) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def stream(repo: str, *, config: str | None = None, split: str = "train"):
    if config:
        return load_dataset(repo, config, split=split, streaming=True)
    return load_dataset(repo, split=split, streaming=True)


def fill_text(path: Path, source: dict, target_bytes: int, seen: set[str]) -> dict:
    total = docs = dup = 0
    ds = stream(source["repo"], config=source.get("config"))
    with path.open("ab") as f:
        for row in ds:
            text = one_line(row.get("text") or row.get("content") or "")
            if len(text) < 160:
                continue
            sig = hashlib.blake2b(text[:8192].encode("utf-8", "ignore"), digest_size=12).hexdigest()
            if sig in seen:
                dup += 1
                continue
            seen.add(sig)
            payload = (text + "\n").encode("utf-8", "ignore")
            remain = target_bytes - total
            if remain <= 0:
                break
            if len(payload) > remain:
                # Avoid a broken UTF-8 tail while still getting close to the byte budget.
                payload = payload[:remain].decode("utf-8", "ignore").encode("utf-8")
            if not payload:
                continue
            f.write(payload)
            total += len(payload)
            docs += 1
            if total >= target_bytes - 4:
                break
    return {"bytes": total, "documents": docs, "duplicates_skipped": dup}


def prepare_main_raw(out: Path, target_bytes: int) -> dict:
    path = out / "main.raw.txt"
    path.unlink(missing_ok=True)
    tr_budget = target_bytes // 2
    en_budget = target_bytes - tr_budget
    seen: set[str] = set()
    try:
        tr = fill_text(path, SOURCES["main_tr"], tr_budget, seen)
    except Exception as exc:
        print(f"v07_data Turkish primary failed: {type(exc).__name__}: {exc}", flush=True)
        tr = fill_text(path, SOURCES["main_tr_fallback"], tr_budget, seen)
    if tr["bytes"] < int(tr_budget * 0.95):
        raise RuntimeError(f"Turkish pretrain underfilled: {tr}")
    en = fill_text(path, SOURCES["main_en"], en_budget, seen)
    if en["bytes"] < int(en_budget * 0.95):
        raise RuntimeError(f"English pretrain underfilled: {en}")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "turkish": tr,
        "english": en,
        "mix": "50% Turkish / 50% English by byte budget",
    }


def coder_document(instruction: str, code: str, source: str) -> str:
    instruction = one_line(instruction) or "Explain, complete, test, or improve the following code."
    code = clean_text(code)
    # IMPORTANT: do not flatten code. v0.6 destroyed indentation with one_line().
    return (
        f"<source>{source}</source>\n"
        f"<instruction>{instruction}</instruction>\n"
        f"<code>\n{code}\n</code>"
    )


def _write_json_text(handle, text: str) -> int:
    raw = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
    handle.write(raw)
    return len(text.encode("utf-8", "ignore"))


def prepare_coder_raw(out: Path, target_bytes: int, sample_bytes: int = 96_000_000) -> dict:
    path = out / "coder_pretrain.jsonl"
    sample = out / "tokenizer_code_sample.txt"
    path.unlink(missing_ok=True)
    sample.unlink(missing_ok=True)
    total = 0
    sample_written = 0
    counts = {"unreal": 0, "python": 0}
    seen: set[str] = set()

    def accept(fh, sfh, instruction, code, source):
        nonlocal total, sample_written
        code = clean_text(code)
        if len(code) < 80:
            return False
        sig = hashlib.blake2b(code[:8192].encode("utf-8", "ignore"), digest_size=12).hexdigest()
        if sig in seen:
            return False
        seen.add(sig)
        doc = coder_document(instruction, code, source)
        n = len(doc.encode("utf-8", "ignore"))
        if total + n > target_bytes and total >= int(target_bytes * .98):
            return False
        _write_json_text(fh, doc)
        total += n
        if sample_written < sample_bytes:
            payload = (doc + "\n").encode("utf-8", "ignore")
            remain = sample_bytes - sample_written
            # Never end tokenizer input in the middle of a multi-byte UTF-8
            # code point. This matters for Turkish characters (ç, ğ, ı, ö, ş, ü).
            chunk = payload[:remain].decode("utf-8", "ignore").encode("utf-8")
            sfh.write(chunk)
            sample_written += len(chunk)
        return True

    with path.open("wb") as fh, sample.open("wb") as sfh:
        unreal_limit = min(64_000_000, target_bytes // 4)
        try:
            for row in stream(SOURCES["coder_unreal"]["repo"]):
                if total >= unreal_limit:
                    break
                code = row.get("code") or ""
                desc = row.get("description") or row.get("className") or ""
                if accept(fh, sfh, desc, code, "unreal-engine-5"):
                    counts["unreal"] += 1
        except Exception as exc:
            print(f"v07_data optional Unreal failed: {type(exc).__name__}: {exc}", flush=True)

        for row in stream(SOURCES["coder_python"]["repo"]):
            if total >= target_bytes:
                break
            code = row.get("code") or row.get("whole_func_string") or row.get("func_code_string") or ""
            desc = row.get("summary") or row.get("docstring") or row.get("func_documentation_string") or ""
            if accept(fh, sfh, desc, code, "codesearchnet-python"):
                counts["python"] += 1

    if total < int(target_bytes * .95):
        raise RuntimeError(f"coder pretrain underfilled: {total}/{target_bytes}")
    return {
        "file": path.name,
        "content_bytes": total,
        "file_bytes": path.stat().st_size,
        "records": counts,
        "tokenizer_sample_bytes": sample.stat().st_size,
    }


def normalize_alpaca(row: dict) -> tuple[str, str] | None:
    row = {str(k).strip(): v for k, v in row.items()}
    instruction = clean_text(row.get("instruction") or row.get("talimat") or row.get("question") or "")
    extra = clean_text(row.get("input") or row.get("giriş") or row.get("giris") or "")
    answer = clean_text(row.get("output") or row.get("çıktı") or row.get("cikti") or row.get("answer") or "")
    if not instruction or not answer:
        return None
    user = instruction if not extra else f"{instruction}\n\nBağlam:\n{extra}"
    return user, answer


def _write_chat(fh, messages: list[dict], source: str, seen: set[str]) -> bool:
    cleaned = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        content = clean_text(msg.get("content") or "")
        if role not in {"system", "user", "assistant", "tool"} or not content:
            continue
        if len(content) > 24000:
            content = content[:24000]
        cleaned.append({"role": role, "content": content})
    if not cleaned or not any(x["role"] == "assistant" for x in cleaned):
        return False
    sig_text = "\n".join(x["role"] + ":" + x["content"] for x in cleaned)
    sig = hashlib.blake2b(sig_text.encode("utf-8", "ignore"), digest_size=12).hexdigest()
    if sig in seen:
        return False
    seen.add(sig)
    fh.write(json.dumps({"messages": cleaned, "source": source}, ensure_ascii=False) + "\n")
    return True


def prepare_main_sft(out: Path, tr_rows_per_source: int, en_rows: int) -> dict:
    path = out / "main_sft.jsonl"
    path.unlink(missing_ok=True)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as fh:
        for key in ("main_sft_tr_alpaca", "main_sft_tr_merve"):
            src = SOURCES[key]
            added = 0
            for row in stream(src["repo"]):
                pair = normalize_alpaca(row)
                if pair is None:
                    continue
                user, answer = pair
                if _write_chat(
                    fh,
                    [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                    src["repo"],
                    seen,
                ):
                    added += 1
                if added >= tr_rows_per_source:
                    break
            counts[key] = added
            print(f"v07_data main_sft {key} rows={added}", flush=True)

        src = SOURCES["main_sft_en_ultrachat"]
        added = 0
        ds = stream(src["repo"], split=src["split"])
        for row in ds:
            messages = row.get("messages") or []
            if _write_chat(fh, messages, src["repo"], seen):
                added += 1
            if added >= en_rows:
                break
        counts["main_sft_en_ultrachat"] = added
        print(f"v07_data main_sft ultrachat rows={added}", flush=True)

    total = sum(counts.values())
    if total < 150_000:
        raise RuntimeError(f"main SFT unexpectedly small: {counts}")
    return {"file": path.name, "rows": total, "sources": counts, "file_bytes": path.stat().st_size}


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def prepare_coder_sft(out: Path, code_rows: int, tool_rows: int, synthetic_rows: int) -> dict:
    path = out / "coder_sft.jsonl"
    path.unlink(missing_ok=True)
    seen: set[str] = set()
    counts = {"code": 0, "xlam_tools": 0, "synthetic_tool_loop": 0}

    with path.open("w", encoding="utf-8") as fh:
        for row in stream(SOURCES["coder_python"]["repo"]):
            code = clean_text(row.get("code") or row.get("whole_func_string") or row.get("func_code_string") or "")
            desc = clean_text(row.get("summary") or row.get("docstring") or row.get("func_documentation_string") or "")
            if len(code) < 80:
                continue
            user = one_line(desc) or "Write or complete this Python function."
            if _write_chat(
                fh,
                [{"role": "user", "content": user}, {"role": "assistant", "content": code}],
                SOURCES["coder_python"]["repo"],
                seen,
            ):
                counts["code"] += 1
            if counts["code"] >= code_rows:
                break

        for row in stream(SOURCES["coder_tools"]["repo"]):
            query = clean_text(row.get("query") or "")
            if not query:
                continue
            try:
                tools = json.loads(row.get("tools") or "[]")
                answers = json.loads(row.get("answers") or "[]")
            except Exception:
                continue
            if not isinstance(tools, list) or not isinstance(answers, list) or not answers:
                continue
            calls = []
            for ans in answers:
                if not isinstance(ans, dict) or not ans.get("name"):
                    continue
                args = ans.get("arguments", ans.get("args", {}))
                if not isinstance(args, dict):
                    continue
                calls.append(
                    "<|tool_call|>"
                    + compact_json({"name": str(ans["name"]), "arguments": args})
                    + "<|tool_end|>"
                )
            if not calls:
                continue
            system = (
                "You are FLM-Coder. Available tools are described by this JSON schema list:\n"
                + compact_json(tools)
                + "\nIf a tool is required, return a structured tool call instead of inventing a result."
            )
            if _write_chat(
                fh,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "\n".join(calls)},
                ],
                SOURCES["coder_tools"]["repo"],
                seen,
            ):
                counts["xlam_tools"] += 1
            if counts["xlam_tools"] >= tool_rows:
                break

        # Small deterministic bilingual traces teach the second half of the
        # protocol: consume a tool result and return a normal final answer.
        for i in range(synthetic_rows):
            a = (i * 17 + 3) % 997
            b = (i * 29 + 11) % 997
            result = a + b
            turkish = (i % 2 == 0)
            system = (
                'Available tools: [{"name":"add","description":"Add two integers",'
                '"parameters":{"type":"object","properties":{"a":{"type":"integer"},'
                '"b":{"type":"integer"}},"required":["a","b"]}}]'
            )
            user = f"{a} ile {b} sayısını topla." if turkish else f"Add {a} and {b}."
            call = "<|tool_call|>" + compact_json({"name": "add", "arguments": {"a": a, "b": b}}) + "<|tool_end|>"
            tool = "<|tool_result|>" + compact_json({"name": "add", "result": result}) + "<|tool_end|>"
            final = f"Sonuç {result}." if turkish else f"The result is {result}."
            if _write_chat(
                fh,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": call},
                    {"role": "tool", "content": tool},
                    {"role": "assistant", "content": final},
                ],
                "synthetic:add-tool-loop-v1",
                seen,
            ):
                counts["synthetic_tool_loop"] += 1

    total = sum(counts.values())
    if counts["xlam_tools"] < min(20_000, tool_rows):
        raise RuntimeError(f"too few xLAM tool rows: {counts}")
    if counts["code"] < min(30_000, code_rows):
        raise RuntimeError(f"too few coder SFT rows: {counts}")
    return {"file": path.name, "rows": total, "sources": counts, "file_bytes": path.stat().st_size}


def train_tokenizer(out: Path, main_raw: Path, code_sample: Path, vocab_size: int) -> dict:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tok.train([str(main_raw), str(code_sample)], trainer)
    path = out / "tokenizer.json"
    tok.save(str(path))
    actual = tok.get_vocab_size()
    if actual < min(vocab_size, 8192):
        raise RuntimeError(f"tokenizer unexpectedly small: {actual}")
    meta = {name: tok.token_to_id(name) for name in SPECIAL}
    if any(v is None for v in meta.values()):
        raise RuntimeError(f"missing special token: {meta}")
    meta["vocab_size"] = actual
    (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"file": path.name, "vocab_size": actual, "special_ids": meta}


def encode_main(tok: Tokenizer, src: Path, dst: Path) -> dict:
    eos = tok.token_to_id("<eos>")
    total = docs = 0
    with src.open("r", encoding="utf-8", errors="ignore") as fh, dst.open("wb") as out:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False).ids + [eos]
            np.asarray(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
            docs += 1
    return {"file": dst.name, "tokens": total, "documents": docs, "bytes": dst.stat().st_size}


def encode_coder_pretrain(tok: Tokenizer, src: Path, dst: Path) -> dict:
    eos = tok.token_to_id("<eos>")
    total = docs = 0
    with src.open("r", encoding="utf-8") as fh, dst.open("wb") as out:
        for line in fh:
            row = json.loads(line)
            ids = tok.encode(row["text"], add_special_tokens=False).ids + [eos]
            np.asarray(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
            docs += 1
    return {"file": dst.name, "tokens": total, "documents": docs, "bytes": dst.stat().st_size}


ROLE_TOKEN = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool_result|>",
}


def encode_chat_sft(tok: Tokenizer, src: Path, token_dst: Path, mask_dst: Path) -> dict:
    ids = {name: tok.token_to_id(name) for name in SPECIAL}
    total = supervised = rows = assistant_turns = 0
    with src.open("r", encoding="utf-8") as fh, token_dst.open("wb") as tf, mask_dst.open("wb") as mf:
        for line in fh:
            row = json.loads(line)
            seq = [ids["<bos>"]]
            mask = [0]
            for msg in row["messages"]:
                role = msg["role"]
                marker = ROLE_TOKEN.get(role)
                if marker is None:
                    continue
                seq.append(ids[marker])
                mask.append(0)
                content_ids = tok.encode(msg["content"], add_special_tokens=False).ids
                seq.extend(content_ids)
                supervise = role == "assistant"
                mask.extend([1 if supervise else 0] * len(content_ids))
                if supervise:
                    seq.append(ids["<|end|>"])
                    mask.append(1)
                    assistant_turns += 1
            seq.append(ids["<eos>"])
            mask.append(1 if row["messages"] and row["messages"][-1]["role"] == "assistant" else 0)
            if not any(mask):
                continue
            np.asarray(seq, dtype=np.uint16).tofile(tf)
            np.asarray(mask, dtype=np.uint8).tofile(mf)
            total += len(seq)
            supervised += sum(mask)
            rows += 1
    if total != np.fromfile(token_dst, dtype=np.uint16).size:
        raise RuntimeError("SFT token serialization mismatch")
    return {
        "tokens_file": token_dst.name,
        "mask_file": mask_dst.name,
        "tokens": total,
        "supervised_tokens": supervised,
        "rows": rows,
        "assistant_turns": assistant_turns,
    }


class ImageShardWriter:
    def __init__(self, out: Path, prefix: str = "cu07_images", max_images: int = 600):
        self.out = out
        self.prefix = prefix
        self.max_images = max_images
        self.index = -1
        self.in_shard = 0
        self.zf: zipfile.ZipFile | None = None
        self.current_name = ""

    def _next(self):
        if self.zf:
            self.zf.close()
        self.index += 1
        self.in_shard = 0
        self.current_name = f"{self.prefix}_{self.index:03d}.zip"
        self.zf = zipfile.ZipFile(self.out / self.current_name, "w", zipfile.ZIP_STORED)

    def add(self, image: Image.Image, member: str) -> tuple[str, str, int, int]:
        if self.zf is None or self.in_shard >= self.max_images:
            self._next()
        im = image.convert("RGB")
        width, height = im.size
        # Keep geometry/aspect ratio; training maps normalized coords linearly.
        max_side = 1280
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            im = im.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        from io import BytesIO
        buf = BytesIO()
        im.save(buf, "JPEG", quality=82, optimize=False)
        assert self.zf is not None
        self.zf.writestr(member, buf.getvalue())
        self.in_shard += 1
        return self.current_name, member, width, height

    def close(self):
        if self.zf:
            self.zf.close()
            self.zf = None


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "pyautogui":
        return f.attr
    return None


def parse_pyautogui_action(source: str, width: int, height: int) -> dict | None:
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return None
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n)]
    if not calls:
        return None

    def kwargs(call):
        return {k.arg: literal(k.value) for k in call.keywords if k.arg}

    def xy(call):
        kw = kwargs(call)
        x = literal(call.args[0]) if len(call.args) > 0 else kw.get("x")
        y = literal(call.args[1]) if len(call.args) > 1 else kw.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        return [
            max(0.0, min(1.0, float(x) / max(1, width - 1))),
            max(0.0, min(1.0, float(y) / max(1, height - 1))),
        ]

    # A moveTo followed by dragTo is one semantically useful drag sample.
    for i, call in enumerate(calls):
        name = _call_name(call)
        if name == "moveTo" and i + 1 < len(calls) and _call_name(calls[i + 1]) == "dragTo":
            a, b = xy(call), xy(calls[i + 1])
            if a and b:
                return {"op": "DRAG", "coord": a, "coord2": b, "payload": ""}

    call = calls[-1]
    name = _call_name(call)
    kw = kwargs(call)

    if name in {"click", "doubleClick", "rightClick", "moveTo"}:
        coord = xy(call)
        if not coord:
            return None
        if name == "moveTo":
            op = "MOVE"
        elif name == "doubleClick" or int(kw.get("clicks") or 1) >= 2:
            op = "DOUBLE_CLICK"
        elif name == "rightClick" or str(kw.get("button") or "left").lower() == "right":
            op = "RIGHT_CLICK"
        else:
            op = "CLICK"
        return {"op": op, "coord": coord, "coord2": None, "payload": ""}

    if name == "dragTo":
        # No reliable start position in a single isolated call.
        return None

    if name in {"write", "typewrite"}:
        value = literal(call.args[0]) if call.args else kw.get("message")
        if isinstance(value, str) and value:
            return {"op": "TYPE", "coord": None, "coord2": None, "payload": value}
        return None

    if name in {"hotkey", "press"}:
        vals = [literal(x) for x in call.args]
        if name == "press" and vals:
            vals = vals[:1]
        keys = [str(x) for x in vals if isinstance(x, (str, int))]
        if keys:
            return {"op": "KEY", "coord": None, "coord2": None, "payload": "+".join(keys)}
        return None

    if name in {"keyDown", "keyUp"}:
        value = literal(call.args[0]) if call.args else None
        if isinstance(value, str):
            return {
                "op": "KEY_DOWN" if name == "keyDown" else "KEY_UP",
                "coord": None, "coord2": None, "payload": value,
            }
        return None

    if name in {"scroll", "hscroll"}:
        value = literal(call.args[0]) if call.args else kw.get("clicks")
        if isinstance(value, (int, float)):
            if name == "hscroll":
                payload = f"{int(value)},0"
            else:
                payload = f"0,{int(value)}"
            return {"op": "SCROLL", "coord": None, "coord2": None, "payload": payload}
        return None

    return None


def prepare_rexx(out: Path, max_trajectories: int) -> dict:
    manifest = out / "cu07_manifest.jsonl"
    manifest.unlink(missing_ok=True)
    writer = ImageShardWriter(out)
    total = trajectories = skipped = successful = 0
    ops: dict[str, int] = {}
    domains: dict[str, int] = {}

    ds = stream(SOURCES["computer_rexx"]["repo"])
    with manifest.open("w", encoding="utf-8") as fh:
        for row in ds:
            if trajectories >= max_trajectories:
                break
            trajectories += 1
            if float(row.get("score") or 0.0) <= 0:
                continue
            successful += 1
            instruction = clean_text(row.get("instruction") or "")
            domain = "rexx:" + one_line(row.get("domain") or "desktop")
            actions = row.get("actions") or []
            images = row.get("screenshots") or []
            statuses = row.get("exe_statuses") or []
            task_id = str(row.get("task_id") or f"traj-{trajectories}")
            for step, (action_src, image) in enumerate(zip(actions, images)):
                if image is None or not hasattr(image, "size"):
                    skipped += 1
                    continue
                if step < len(statuses) and str(statuses[step]).lower() not in {"success", "none", ""}:
                    skipped += 1
                    continue
                width, height = image.size
                parsed = parse_pyautogui_action(action_src, width, height)
                if parsed is None:
                    skipped += 1
                    continue
                member = f"{task_id}_{step:04d}.jpg".replace("/", "_")
                archive, member, _, _ = writer.add(image, member)
                record = {
                    "archive": archive,
                    "image": member,
                    "task": instruction,
                    "operation": parsed["op"],
                    "coord": parsed["coord"],
                    "coord2": parsed["coord2"],
                    "payload": parsed["payload"],
                    "domain": domain,
                    "source": SOURCES["computer_rexx"]["repo"],
                    "episode_id": task_id,
                    "step": step,
                    "action_source": str(action_src)[:2000],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                ops[parsed["op"]] = ops.get(parsed["op"], 0) + 1
                domains[domain] = domains.get(domain, 0) + 1
    writer.close()
    if total < 500:
        raise RuntimeError(f"too few parsed REXX input events: {total}")
    return {
        "file": manifest.name,
        "trajectories_scanned": trajectories,
        "successful_trajectories": successful,
        "examples": total,
        "skipped_actions": skipped,
        "ops": ops,
        "domains": domains,
    }


def copy_prefix(src: Path, dst: Path, budget: int) -> int:
    """Copy a bounded tokenizer sample while preserving valid UTF-8.

    Byte-budget truncation can land inside a Turkish/non-ASCII code point.
    Tokenizers requires valid UTF-8, so use an incremental decoder and drop
    only an incomplete/invalid tail instead of corrupting the whole sample.
    """
    import codecs

    written = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    with src.open("rb") as i, dst.open("wb") as o:
        while written < budget:
            chunk = i.read(min(1024 * 1024, budget - written))
            if not chunk:
                break
            clean = decoder.decode(chunk, final=False).encode("utf-8")
            if len(clean) > budget - written:
                clean = clean[: budget - written].decode("utf-8", "ignore").encode("utf-8")
            o.write(clean)
            written += len(clean)
        if written < budget:
            tail = decoder.decode(b"", final=True).encode("utf-8")
            if tail:
                tail = tail[: budget - written].decode("utf-8", "ignore").encode("utf-8")
                o.write(tail)
                written += len(tail)

    # Fail here, close to the source of the problem, rather than minutes later
    # inside tokenizers with an opaque Rust "stream did not contain UTF-8".
    dst.read_text(encoding="utf-8")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prepared_v07")
    ap.add_argument("--owner", required=True)
    ap.add_argument("--main-bytes", type=int, default=1_024_000_000)
    ap.add_argument("--coder-bytes", type=int, default=256_000_000)
    ap.add_argument("--vocab-size", type=int, default=16_384)
    ap.add_argument("--tr-sft-per-source", type=int, default=60_000)
    ap.add_argument("--en-sft-rows", type=int, default=120_000)
    ap.add_argument("--coder-sft-code-rows", type=int, default=60_000)
    ap.add_argument("--tool-sft-rows", type=int, default=60_000)
    ap.add_argument("--synthetic-tool-rows", type=int, default=6_000)
    ap.add_argument("--rexx-trajectories", type=int, default=320)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()

    stats = {}
    print("v07_data main_raw_start", flush=True)
    stats["main_raw"] = prepare_main_raw(out, args.main_bytes)
    print("v07_data main_raw_done", json.dumps(stats["main_raw"]), flush=True)

    print("v07_data coder_raw_start", flush=True)
    stats["coder_raw"] = prepare_coder_raw(out, args.coder_bytes)
    print("v07_data coder_raw_done", json.dumps(stats["coder_raw"]), flush=True)

    print("v07_data main_sft_start", flush=True)
    stats["main_sft_raw"] = prepare_main_sft(out, args.tr_sft_per_source, args.en_sft_rows)
    print("v07_data main_sft_done", json.dumps(stats["main_sft_raw"]), flush=True)

    print("v07_data coder_sft_start", flush=True)
    stats["coder_sft_raw"] = prepare_coder_sft(
        out, args.coder_sft_code_rows, args.tool_sft_rows, args.synthetic_tool_rows
    )
    print("v07_data coder_sft_done", json.dumps(stats["coder_sft_raw"]), flush=True)

    print("v07_data rexx_start", flush=True)
    stats["computer_rexx"] = prepare_rexx(out, args.rexx_trajectories)
    print("v07_data rexx_done", json.dumps(stats["computer_rexx"]), flush=True)

    # Train the vocabulary on a bounded sample so a 1+ GB corpus does not make
    # tokenizer training itself the bottleneck.
    main_sample = out / "tokenizer_main_sample.txt"
    copy_prefix(out / "main.raw.txt", main_sample, 256_000_000)
    stats["tokenizer"] = train_tokenizer(
        out, main_sample, out / "tokenizer_code_sample.txt", args.vocab_size
    )
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    print("v07_data tokenizer_done", json.dumps(stats["tokenizer"]), flush=True)

    stats["main"] = encode_main(tok, out / "main.raw.txt", out / "main_train.u16")
    stats["coder"] = encode_coder_pretrain(tok, out / "coder_pretrain.jsonl", out / "coder_train.u16")
    stats["main_sft"] = encode_chat_sft(
        tok, out / "main_sft.jsonl",
        out / "main_sft_tokens.u16", out / "main_sft_mask.u8",
    )
    stats["coder_sft"] = encode_chat_sft(
        tok, out / "coder_sft.jsonl",
        out / "coder_sft_tokens.u16", out / "coder_sft_mask.u8",
    )

    # Remove large CPU-only preprocessing intermediates before Kaggle upload.
    for name in (
        "main.raw.txt", "coder_pretrain.jsonl", "tokenizer_main_sample.txt",
        "tokenizer_code_sample.txt", "main_sft.jsonl", "coder_sft.jsonl",
    ):
        (out / name).unlink(missing_ok=True)

    manifest = {
        "pipeline_version": "v0.7",
        "owner": args.owner,
        "sources": SOURCES,
        "stats": stats,
        "format": {
            "main_train.u16": "uint16 BPE token IDs",
            "coder_train.u16": "uint16 BPE token IDs; source code formatting preserved",
            "main_sft_*": "packed chat SFT IDs + assistant-token loss mask",
            "coder_sft_*": "packed code/tool SFT IDs + assistant-token loss mask",
            "cu07_manifest.jsonl": "one executable desktop input action per screenshot",
        },
    }
    (out / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "title": "FLM v0.7 Bilingual Tool Desktop Data",
        "id": f"{args.owner}/flm-hf-v07",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("V07_DATA_COMPLETE", json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
