#!/usr/bin/env python3
from __future__ import annotations

import json
from flm.core_agent import TrainingFreeAgent

agent=TrainingFreeAgent()
rows=[]

def run(question, expected=None):
    try:
        answer=agent.ask(question)
        ok=(expected is None or expected in answer)
        rows.append({"question":question,"answer":answer,"ok":ok})
    except Exception as exc:
        rows.append({"question":question,"answer":f"{type(exc).__name__}: {exc}","ok":False})

run("Mert bir bilgisayardır.", "Bilgi kaydedildi")
run("Mert bir bilgisayar mıdır?", "Evet")
run("Her memeli sıcakkanlıdır.", "Bilgi kaydedildi")
run("Ali bir memelidir.", "Bilgi kaydedildi")
run("Ali sıcakkanlı mıdır?", "Evet")
run("Ada bir kedidir.", "Bilgi kaydedildi")
run("Ada bir memelidir.", "Bilgi kaydedildi")
memory_answers=[agent.ask("Ada hakkında ne biliyorsun?") for _ in range(3)]
rows.append({
    "question":"Ada hakkında ne biliyorsun? (3 kez)",
    "answer":" || ".join(memory_answers),
    "ok":all("Ada" in x and "kedi" in x and "memeli" in x for x in memory_answers)
        and len(set(memory_answers))==3,
})
identity_answers=[agent.ask("Sen kimsin?") for _ in range(8)]
rows.append({
    "question":"Sen kimsin? (8 kez)",
    "answer":" || ".join(identity_answers),
    "ok":all("FLM" in x for x in identity_answers)
        and all(any(k in x.casefold() for k in ("yapay zek", "ai ", "asistan")) for x in identity_answers)
        and len(set(identity_answers))==8,
})
run("Hesapla: 27*14", "378")
compare_answer=agent.ask("17 mi büyük 9 mu?")
rows.append({
    "question":"17 mi büyük 9 mu?",
    "answer":compare_answer,
    "ok":"17" in compare_answer and "9" in compare_answer
        and any(x in compare_answer.casefold() for x in ("büyük","daha büyük","aşıyor","aşar","üstünde")),
})
sort_answer=agent.ask("Şunları küçükten büyüğe sırala: 9, 3, 7, 1.")
rows.append({
    "question":"Şunları küçükten büyüğe sırala: 9, 3, 7, 1.",
    "answer":sort_answer,
    "ok":"1, 3, 7, 9" in sort_answer,
})
run("""Bu Python kodunu analiz et:
```python
def kare(x):
    return print(abs(x*x))
```
""", "kare")
run("""Bu Python kodunu analiz et:
```python
def bozuk(x)
    return x+1
```
""", "SyntaxError")

print("FLM_TRAINING_FREE_EVAL="+json.dumps(rows,ensure_ascii=False))
if not all(x["ok"] for x in rows):
    raise SystemExit(1)
