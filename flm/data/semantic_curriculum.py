from __future__ import annotations

import json
import random
from pathlib import Path


TR_NAMES=("Ali","Ayşe","Mehmet","Zeynep","Deniz","Ece","Mert","Selin")
EN_NAMES=("Alice","Bob","Charlie","Diana","Ethan","Grace","Henry","Iris")
TYPES=(
    ("kedi","Cat"),("köpek","Dog"),("kuş","Bird"),("memeli","Mammal"),
    ("araç","Vehicle"),("bilgisayar","Computer"),("program","Program"),("dosya","File"),
)
RELATIONS=(
    ("daha uzundur","is taller than","TallerThan"),
    ("daha hızlıdır","is faster than","FasterThan"),
    ("içindedir","is in","LocatedIn"),
)


def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"),sort_keys=True)


def semantic_ir(facts=None,rules=None,queries=None):
    return {
        "facts":facts or [],
        "rules":rules or [],
        "queries":queries or [],
        "metadata":{"source":"synthetic-semantic-curriculum-v1"},
    }


def atom(pred,*args):
    return {"predicate":pred,"args":list(args)}


def messages_for(i:int):
    tr=(i%2==0)
    names=TR_NAMES if tr else EN_NAMES
    a=names[i%len(names)]
    b=names[(i*3+1)%len(names)]
    c=names[(i*5+2)%len(names)]
    mode=i%4

    if mode==0:
        tr_type,en_type,sym=TYPES[(i//4)%len(TYPES)]
        sentence=f"{a} bir {tr_type}dir." if tr else f"{a} is a {en_type.lower()}."
        prog=semantic_ir(facts=[atom("IsA",a,sym)])
        final=f"{a} bir {tr_type} olarak kaydedildi." if tr else f"{a} was recorded as a {en_type.lower()}."
    elif mode==1:
        tr_rel,en_rel,pred=RELATIONS[(i//4)%len(RELATIONS)]
        sentence=f"{a} {b}'den {tr_rel}." if tr else f"{a} {en_rel} {b}."
        prog=semantic_ir(facts=[atom(pred,a,b)])
        final="İlişki kaydedildi." if tr else "The relation was recorded."
    elif mode==2:
        # Two facts + a reusable transitive rule + a query.
        sentence=(
            f"{a}, {b}'den uzundur. {b}, {c}'den uzundur. {a}, {c}'den uzun mudur?"
            if tr else
            f"{a} is taller than {b}. {b} is taller than {c}. Is {a} taller than {c}?"
        )
        rule={
            "name":"taller_transitive",
            "premises":[atom("TallerThan","?x","?y"),atom("TallerThan","?y","?z")],
            "conclusion":atom("TallerThan","?x","?z"),
        }
        prog=semantic_ir(
            facts=[atom("TallerThan",a,b),atom("TallerThan",b,c)],
            rules=[rule],
            queries=[{"atom":atom("TallerThan",a,c)}],
        )
        final="Evet." if tr else "Yes."
    else:
        sentence=(
            f"Bir memeli sıcakkanlıdır. {a} bir memelidir. {a} sıcakkanlı mıdır?"
            if tr else
            f"Every mammal is warm-blooded. {a} is a mammal. Is {a} warm-blooded?"
        )
        rule={
            "name":"mammal_warm_blooded",
            "premises":[atom("IsA","?x","Mammal")],
            "conclusion":atom("WarmBlooded","?x"),
        }
        prog=semantic_ir(
            facts=[atom("IsA",a,"Mammal")],
            rules=[rule],
            queries=[{"atom":atom("WarmBlooded",a)}],
        )
        final="Evet, sıcakkanlıdır." if tr else "Yes, it is warm-blooded."

    ir="<|semantic_ir|>"+compact(prog)+"<|semantic_end|>"
    core="<|core_result|>"+compact({"queries":[{"answer":True}]})+"<|core_end|>"
    system=(
        "You are the single FLM Interface Transformer. Convert user language into "
        "FLM Semantic IR. Do not perform hidden reasoning when FLM Core can execute it."
    )
    return [
        {"role":"system","content":system},
        {"role":"user","content":sentence},
        {"role":"assistant","content":ir},
        {"role":"tool","content":core},
        {"role":"assistant","content":"<|final|>\n"+final},
    ]


def append_semantic_curriculum(path:Path,rows:int=120_000,seed:int=7071)->dict:
    rng=random.Random(seed)
    indices=list(range(rows))
    rng.shuffle(indices)
    with path.open("a",encoding="utf-8") as fh:
        for i in indices:
            fh.write(json.dumps({
                "messages":messages_for(i),
                "source":"synthetic:flm-semantic-curriculum-v1",
            },ensure_ascii=False)+"\n")
    return {"rows":rows,"source":"synthetic:flm-semantic-curriculum-v1"}
