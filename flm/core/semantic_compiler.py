from __future__ import annotations

import json
import re
import unicodedata

from .semantic_ir import Atom, Program, Query, Rule


_WS=re.compile(r"\s+")


def _norm(text:str)->str:
    text=unicodedata.normalize("NFKC",str(text or "")).strip()
    return _WS.sub(" ",text)


class SemanticCompiler:
    """Deterministic fallback compiler for a small explicit TR/EN grammar.

    The InterfaceTransformer handles open-ended language. This compiler keeps a
    fully training-free path for canonical statements, rules and queries.
    """

    COPULA_PATTERNS=(
        re.compile(r"^(?P<a>.+?) bir (?P<b>[^.?!]+?)(?:dır|dir|dur|dür|tır|tir|tur|tür)?[.]?$",re.I),
        re.compile(r"^(?P<a>.+?) is an? (?P<b>[^.?!]+)[.]?$",re.I),
    )
    REL_PATTERNS=(
        (re.compile(r"^(?P<a>.+?) (?P<b>.+?)['’]den daha uzun(?:dur)?[.]?$",re.I),"TallerThan"),
        (re.compile(r"^(?P<a>.+?) is taller than (?P<b>.+?)[.]?$",re.I),"TallerThan"),
        (re.compile(r"^(?P<a>.+?) (?P<b>.+?) içinde(?:dir)?[.]?$",re.I),"LocatedIn"),
        (re.compile(r"^(?P<a>.+?) is in (?P<b>.+?)[.]?$",re.I),"LocatedIn"),
    )

    @staticmethod
    def symbol(value:str)->str:
        value=_norm(value).strip(" .?!")
        value=re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü_]+","_",value)
        value=value.strip("_")
        if not value:
            raise ValueError("empty semantic symbol")
        if value[0].isdigit():
            value="N_"+value
        return value

    def compile_json(self,raw:str)->Program:
        raw=raw.strip()
        if raw.startswith("<|semantic_ir|>"):
            raw=raw[len("<|semantic_ir|>"):]
        if raw.endswith("<|semantic_end|>"):
            raw=raw[:-len("<|semantic_end|>")]
        return Program.from_json(raw.strip())

    def compile_canonical(self,text:str)->Program:
        text=_norm(text)
        for pat in self.COPULA_PATTERNS:
            m=pat.match(text)
            if m:
                a=self.symbol(m.group("a"))
                b=self.symbol(m.group("b"))
                return Program(facts=[Atom("IsA",(a,b))])
        for pat,pred in self.REL_PATTERNS:
            m=pat.match(text)
            if m:
                return Program(facts=[Atom(pred,(self.symbol(m.group("a")),self.symbol(m.group("b"))))])
        raise ValueError("sentence is outside deterministic semantic grammar")

    def query_canonical(self,text:str)->Program:
        text=_norm(text).rstrip("?.!")
        patterns=(
            (re.compile(r"^(?P<a>.+?) bir (?P<b>.+?) mi(?:dir)?$",re.I),"IsA"),
            (re.compile(r"^is (?P<a>.+?) an? (?P<b>.+?)$",re.I),"IsA"),
            (re.compile(r"^(?P<a>.+?) (?P<b>.+?)['’]den daha uzun mu$",re.I),"TallerThan"),
            (re.compile(r"^is (?P<a>.+?) taller than (?P<b>.+?)$",re.I),"TallerThan"),
        )
        for pat,pred in patterns:
            m=pat.match(text)
            if m:
                return Program(queries=[Query(Atom(pred,(self.symbol(m.group("a")),self.symbol(m.group("b")))))])
        raise ValueError("question is outside deterministic semantic grammar")
