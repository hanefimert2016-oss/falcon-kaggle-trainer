#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def letter_ratio(text: str) -> float:
    chars=[c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    letters=sum(c.isalpha() or c.isdigit() for c in chars)
    return letters/len(chars)


def punctuation_ratio(text: str) -> float:
    chars=[c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    punct=sum(not (c.isalpha() or c.isdigit()) for c in chars)
    return punct/len(chars)


def repeated_token_ratio(text: str) -> float:
    toks=re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    if len(toks)<4:
        return 0.0
    from collections import Counter
    c=Counter(toks)
    return max(c.values())/len(toks)


def validate(samples: list[dict]) -> list[str]:
    problems=[]
    by_prompt={str(x.get("prompt","")):str(x.get("output","")).strip() for x in samples}
    required=[
        "Merhaba, nasılsın?",
        "Türkiye'nin başkenti neresidir?",
        "Bugün kendimi biraz yorgun hissediyorum. Bana kısa bir öneri ver.",
        "Hello! How are you today?",
    ]
    for p in required:
        if p not in by_prompt:
            problems.append(f"missing prompt: {p}")
            continue
        out=by_prompt[p]
        if len(out)<12:
            problems.append(f"too short: {p!r} -> {out!r}")
        if "\ufffd" in out:
            problems.append(f"replacement chars: {p!r}")
        if letter_ratio(out)<0.60:
            problems.append(f"low letter ratio {letter_ratio(out):.2f}: {p!r} -> {out[:120]!r}")
        if punctuation_ratio(out)>0.24:
            problems.append(f"punctuation spam {punctuation_ratio(out):.2f}: {p!r} -> {out[:120]!r}")
        if repeated_token_ratio(out)>0.42:
            problems.append(f"token repetition {repeated_token_ratio(out):.2f}: {p!r} -> {out[:120]!r}")

    capital=by_prompt.get("Türkiye'nin başkenti neresidir?","").lower()
    if "ankara" not in capital:
        problems.append("Turkish capital factual smoke check failed: expected 'Ankara'")

    greeting=by_prompt.get("Merhaba, nasılsın?","").lower()
    if len(set(re.findall(r"[a-zçğıöşü]+",greeting)))<4:
        problems.append("Turkish greeting lexical-diversity smoke check failed")

    english=by_prompt.get("Hello! How are you today?","").lower()
    if len(set(re.findall(r"[a-z]+",english)))<4:
        problems.append("English greeting lexical-diversity smoke check failed")
    return problems


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("samples",help="quality_samples.json")
    args=ap.parse_args()
    path=Path(args.samples)
    samples=json.loads(path.read_text(encoding="utf-8"))
    problems=validate(samples)
    if problems:
        print("V06_QUALITY_FAIL")
        for p in problems:
            print("-",p)
        return 2
    print("V06_QUALITY_PASS")
    for item in samples:
        print(item["prompt"])
        print("  ->",item["output"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
