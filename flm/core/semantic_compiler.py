from __future__ import annotations

import json
import re
import unicodedata

from .semantic_ir import Atom, Operation, Program, Query, Rule


_WS=re.compile(r"\s+")


TYPE_ALIASES={
    "kedi":"Cat","cat":"Cat",
    "köpek":"Dog","kopek":"Dog","dog":"Dog",
    "kuş":"Bird","kus":"Bird","bird":"Bird",
    "memeli":"Mammal","mammal":"Mammal",
    "araç":"Vehicle","arac":"Vehicle","vehicle":"Vehicle",
    "bilgisayar":"Computer","computer":"Computer",
    "program":"Program",
    "dosya":"File","file":"File",
    "insan":"Human","human":"Human","person":"Human",
    "hayvan":"Animal","animal":"Animal",
}
PREDICATE_ALIASES={
    "sıcakkanlı":"WarmBlooded","sicakkanli":"WarmBlooded","warm-blooded":"WarmBlooded",
}


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
        key=value.casefold()
        if key in TYPE_ALIASES:
            return TYPE_ALIASES[key]
        if key in PREDICATE_ALIASES:
            return PREDICATE_ALIASES[key]
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

    def rule_canonical(self,text:str)->Program:
        text=_norm(text)
        patterns=(
            re.compile(
                r"^her (?P<a>[^.?!]+?) (?P<b>sıcakkanlı|sicakkanli)(?:dır|dir|dur|dür|tır|tir|tur|tür)?[.]?$",
                re.I,
            ),
            re.compile(r"^every (?P<a>[^.?!]+?) is (?P<b>warm-blooded)[.]?$",re.I),
        )
        for pat in patterns:
            m=pat.match(text)
            if m:
                src=self.symbol(m.group("a"))
                pred=self.symbol(m.group("b"))
                return Program(rules=[Rule(
                    (Atom("IsA",("?x",src)),),
                    Atom(pred,("?x",)),
                    name=f"{src}_implies_{pred}",
                )])
        raise ValueError("sentence is outside deterministic rule grammar")

    def operation_canonical(self,text:str)->Program:
        raw=unicodedata.normalize("NFKC",str(text or "")).strip()
        normalized=_norm(raw)

        # Arithmetic can bypass the neural interface completely.
        patterns=(
            re.compile(r"^(?P<expr>[0-9+\-*/%(). ^]+) işlemini hesapla[.]?$",re.I),
            re.compile(r"^hesapla[: ]+(?P<expr>[0-9+\-*/%(). ^]+)[.]?$",re.I),
            re.compile(r"^calculate[: ]+(?P<expr>[0-9+\-*/%(). ^]+)[.]?$",re.I),
        )
        for pat in patterns:
            m=pat.match(normalized)
            if m:
                expr=m.group("expr").strip().replace("^","**")
                if not expr:
                    break
                return Program(operations=[Operation("ARITHMETIC",{"expression":expr})])

        # Canonical code-analysis requests keep source formatting intact.
        fenced=re.search(r"```(?P<lang>python|py)?\s*\n(?P<src>.*?)```",raw,re.I|re.S)
        wants_code=bool(re.search(
            r"(python|kod|code).*(analiz|incele|ne yapıyor|ne yapiyor|analyze|inspect|what does)",
            normalized,re.I,
        ))
        if fenced and wants_code:
            lang=(fenced.group("lang") or "python").lower()
            if lang=="py":
                lang="python"
            source=fenced.group("src").rstrip()+"\n"
            return Program(operations=[Operation(
                "ANALYZE_CODE",{"language":lang,"source":source}
            )])

        raise ValueError("sentence is outside deterministic operation grammar")

    def compile_any(self,text:str)->Program:
        stripped=str(text or "").strip()
        if stripped.startswith("{") or stripped.startswith("<|semantic_ir|>"):
            try:
                return self.compile_json(stripped)
            except Exception:
                pass
        for fn in (
            self.query_canonical,
            self.rule_canonical,
            self.compile_canonical,
            self.operation_canonical,
        ):
            try:
                return fn(stripped)
            except ValueError:
                continue
        raise ValueError("input is outside deterministic semantic grammar")

    def query_canonical(self,text:str)->Program:
        text=_norm(text).rstrip("?.!")
        patterns=(
            (re.compile(r"^(?P<a>.+?) bir (?P<b>.+?) m[ıiuü](?:dır|dir|dur|dür)?$",re.I),"IsA"),
            (re.compile(r"^is (?P<a>.+?) an? (?P<b>.+?)$",re.I),"IsA"),
            (re.compile(r"^(?P<a>.+?) (?P<b>.+?)['’]den daha uzun mu$",re.I),"TallerThan"),
            (re.compile(r"^is (?P<a>.+?) taller than (?P<b>.+?)$",re.I),"TallerThan"),
        )
        unary_patterns=(
            (re.compile(r"^(?P<a>.+?) sıcakkanlı m[ıiuü](?:dır|dir)?$",re.I),"WarmBlooded"),
            (re.compile(r"^is (?P<a>.+?) warm-blooded$",re.I),"WarmBlooded"),
        )
        for pat,pred in unary_patterns:
            m=pat.match(text)
            if m:
                return Program(queries=[Query(Atom(pred,(self.symbol(m.group("a")),)))])
        for pat,pred in patterns:
            m=pat.match(text)
            if m:
                return Program(queries=[Query(Atom(pred,(self.symbol(m.group("a")),self.symbol(m.group("b")))))])
        raise ValueError("question is outside deterministic semantic grammar")
