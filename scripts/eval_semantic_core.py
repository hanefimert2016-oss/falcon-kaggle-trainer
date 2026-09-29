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
identity_answers=[agent.ask("Sen kimsin?") for _ in range(4)]
rows.append({
    "question":"Sen kimsin? (4 kez)",
    "answer":" || ".join(identity_answers),
    "ok":all("FLM" in x for x in identity_answers)
        and all(any(k in x for k in ("Ben","Adım","FLM'im","Bana FLM")) for x in identity_answers)
        and len(set(identity_answers))>=3,
})
run("Hesapla: 27*14", "378")
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
