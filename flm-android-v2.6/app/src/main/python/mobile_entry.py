import ast
import json
import math
import operator
import re
import sys
import threading

_lock = threading.RLock()
_turn = 0
_last_answer = ""

TR_ASCII = str.maketrans({"ı":"i","ç":"c","ğ":"g","ö":"o","ş":"s","ü":"u"})

BASIC_FACTS = {
    "turkiye baskenti": "Türkiye'nin başkenti Ankara'dır.",
    "türkiye başkenti": "Türkiye'nin başkenti Ankara'dır.",
    "http 404": "HTTP 404, istenen kaynağın bulunamadığını belirten Not Found durum kodudur.",
    "python": "Python; genel amaçlı, okunabilir sözdizimine sahip bir programlama dilidir.",
    "javascript": "JavaScript; özellikle web tarayıcılarında ve Node.js ortamında kullanılan bir programlama dilidir.",
    "flm tokenizer": "FLM tokenizer tarafında SentencePiece unigram yaklaşımını hedefler.",
}

def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower()).translate(TR_ASCII)

def _safe_number_expr(expr):
    expr = expr.replace("^", "**").replace(",", ".")
    tree = ast.parse(expr, mode="eval")
    binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binops:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 12:
                raise ValueError("power too large")
            return binops[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unops:
            return unops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return ev(tree)

def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)

def _math_answer(text):
    n = _norm(text)

    if re.search(r"\b(?:ebob|gcd)\b", n):
        nums = [int(x) for x in re.findall(r"-?\d+", n)]
        if len(nums) >= 2:
            return f"Sonuç: {math.gcd(nums[0], nums[1])}"

    if re.search(r"\b(?:ekok|lcm)\b", n):
        nums = [int(x) for x in re.findall(r"-?\d+", n)]
        if len(nums) >= 2:
            return f"Sonuç: {math.lcm(nums[0], nums[1])}"

    m = re.search(r"(?:karekok|sqrt)\s*\(?\s*(-?\d+(?:[.,]\d+)?)", n)
    if m:
        x = float(m.group(1).replace(",", "."))
        if x < 0:
            return "Gerçel sayılarda negatif bir sayının karekökü tanımlı değildir."
        return f"Sonuç: {_fmt(math.sqrt(x))}"

    # "Bir sayının %20'si 50 ise"
    m = re.search(r"(?:bir sayinin\s*)?%\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:'?(?:si|i))?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:ise|olursa)", n)
    if m:
        p = float(m.group(1).replace(",", "."))
        y = float(m.group(2).replace(",", "."))
        if p == 0:
            return "Yüzde değeri 0 olduğunda başlangıç sayısı tekil olarak belirlenemez."
        return f"Sonuç: {_fmt(y * 100.0 / p)}"

    # "200'ün %15'i"
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:'?(?:un|in|nin|nun))?\s*%\s*([0-9]+(?:[.,][0-9]+)?)", n)
    if m:
        x = float(m.group(1).replace(",", "."))
        p = float(m.group(2).replace(",", "."))
        return f"Sonuç: {_fmt(x * p / 100.0)}"

    # "50, 200'ün yüzde kaçı"
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*,?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:'?(?:un|in|nin|nun))?\s*yuzde\s*kaci", n)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        if b == 0:
            return "Sıfıra göre yüzde oranı hesaplanamaz."
        return f"Sonuç: %{_fmt(a * 100.0 / b)}"

    # Ax + B = C or Ax+B = Dx+E
    eq = n.replace(" ", "").replace("−", "-")
    m = re.search(r"(-?\d*(?:[.,]\d+)?)x([+-]\d+(?:[.,]\d+)?)?=(-?\d*(?:[.,]\d+)?)x?([+-]\d+(?:[.,]\d+)?)?$", eq)
    if m and "x" in eq:
        def coef(s, default=0.0):
            if s is None or s == "":
                return default
            if s == "-":
                return -1.0
            return float(s.replace(",", "."))
        left_a = coef(m.group(1), 1.0)
        left_b = coef(m.group(2), 0.0)
        right_raw = m.group(3)
        right_tail = m.group(4)
        right_has_x = "x" in eq.split("=", 1)[1]
        if right_has_x:
            right_a = coef(right_raw, 1.0)
            right_b = coef(right_tail, 0.0)
        else:
            right_a = 0.0
            right_b = coef(right_raw, 0.0)
        den = left_a - right_a
        if abs(den) < 1e-12:
            return "Denklemin tek bir çözümü yok."
        x = (right_b - left_b) / den
        return f"Sonuç: x = {_fmt(x)}"

    # Quadratic ax^2+bx+c=0
    q = eq.replace("x²", "x^2")
    m = re.search(r"([+-]?\d*)x\^2([+-]\d*)x([+-]\d+)=0$", q)
    if m:
        def qc(s, default):
            if s in ("", "+"):
                return default
            if s == "-":
                return -default
            return float(s)
        a = qc(m.group(1), 1.0)
        b = qc(m.group(2), 1.0)
        c = float(m.group(3))
        d = b*b - 4*a*c
        if d < 0:
            return "Bu denklemin gerçel kökü yok."
        if abs(d) < 1e-12:
            return f"Sonuç: x = {_fmt(-b/(2*a))}"
        r = math.sqrt(d)
        x1 = (-b-r)/(2*a)
        x2 = (-b+r)/(2*a)
        vals = sorted([x1, x2])
        return f"Sonuç: x = {_fmt(vals[0])}, {_fmt(vals[1])}"

    # Basic arithmetic: require operator and numbers.
    candidate = n
    candidate = re.sub(r"\b(?:kac|eder|sonuc|nedir|hesapla)\b", "", candidate)
    candidate = candidate.replace("carpi", "*").replace("bolu", "/").replace("arti", "+").replace("eksi", "-")
    candidate = re.sub(r"[^0-9+\-*/%().,^ ]", "", candidate).strip()
    if candidate and re.search(r"\d", candidate) and re.search(r"[+\-*/%^]", candidate):
        try:
            return f"Sonuç: {_fmt(_safe_number_expr(candidate))}"
        except ZeroDivisionError:
            return "Sıfıra bölme tanımsızdır."
        except Exception:
            pass
    return None

def _code_answer(text):
    n = _norm(text)
    if not re.search(r"\b(python|javascript|js|html|css|kod|fonksiyon|function|script|program)\b", n):
        return None

    if "faktoriyel" in n or "factorial" in n:
        return """```python
def faktoriyel(n):
    if n < 0:
        raise ValueError("n negatif olamaz")
    sonuc = 1
    for i in range(2, n + 1):
        sonuc *= i
    return sonuc
```"""

    if ("topla" in n or "toplam" in n) and "python" in n:
        return """```python
def topla(a, b):
    return a + b
```"""

    if "async" in n and ("http" in n or "istek" in n):
        return """```python
import asyncio
from urllib.request import urlopen

async def http_get_async(url, timeout=10):
    def fetch():
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    return await asyncio.to_thread(fetch)
```"""

    if "json" in n and ("oku" in n or "read" in n):
        return """```python
import json
from pathlib import Path

def json_oku(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
```"""

    if "csv" in n and ("oku" in n or "read" in n):
        return """```python
import csv

def csv_oku(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
```"""

    if ("javascript" in n or re.search(r"\bjs\b", n)) and ("carp" in n or "multiply" in n):
        return """```javascript
function multiply(a, b) {
  return a * b;
}
```"""

    if "html" in n and ("oyun" in n or "game" in n):
        return """Tek dosyalık HTML oyun isteklerini FLM'nin masaüstü sürümündeki Game IR daha ayrıntılı üretir. Android sürümünde de kodu oluşturabilirim; proje dosyası kaydetme arayüzünü bir sonraki mobil turda ekliyorum."""

    return """Kod isteğini anladım. Android v2.6'da temel Python/JavaScript üretimi yerel çalışıyor; çok dosyalı proje üretimi için Code Mode ekranını ayrıca ekleyeceğim."""

def _computer_plan(text):
    raw = (text or "").strip()
    n = _norm(raw)
    if re.search(r"\b(python|javascript|html|css|kod|fonksiyon|script|program|oyun)\b", n):
        return None

    cues = ("ac", "aç", "tikla", "tıkla", "dokun", "yaz", "gir", "bas", "scroll", "kaydir", "kaydır", "open", "click", "tap", "type", "press", "home", "geri")
    if not any(c in n for c in cues):
        return None

    chunks = [x.strip(" .") for x in re.split(r"\b(?:sonra|ardindan|ardından|then|and then)\b|;", raw, flags=re.I) if x.strip(" .")]
    if len(chunks) == 1 and "," in raw:
        chunks = [x.strip(" .") for x in raw.split(",") if x.strip(" .")]

    actions = []
    aliases = {
        "tarayici": "chrome", "browser": "chrome", "chrome": "chrome",
        "youtube": "youtube", "gmail": "gmail", "ayarlar": "settings",
        "settings": "settings", "fotograflar": "photos", "photos": "photos",
        "play store": "play store", "playstore": "play store",
    }

    for part in chunks:
        q = _norm(part)

        url = re.search(r"(?:https?://[^\s]+|(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)", part, re.I)
        if url and any(k in q for k in ("git", "ac", "aç", "open", "adres", "site")):
            u = url.group(0).rstrip(".,)")
            if not re.match(r"https?://", u, re.I):
                u = "https://" + u
            actions.append({"action": "open_url", "url": u, "target": u})
            continue

        if re.search(r"\b(geri|back)\b", q):
            actions.append({"action": "press_key", "key": "BACK"})
            continue
        if re.search(r"\b(ana ekran|home)\b", q):
            actions.append({"action": "press_key", "key": "HOME"})
            continue

        m = re.search(r"(?:ac|aç|open)\s+(.+?)(?:\s+uygulamasini|\s+uygulamasını)?$", q)
        if not m:
            m = re.search(r"^(.+?)(?:\s+uygulamasini|\s+uygulamasını)?\s+(?:ac|aç)$", q)
        if m:
            target = m.group(1).strip(" .'\"")
            target = re.sub(r"['’]?(?:u|ü|i|ı|yi|yı|yu|yü)$", "", target)
            target = aliases.get(target, target)
            actions.append({"action": "open_app", "target": target})
            continue

        m = re.search(r"(?:tikla|tıkla|click|tap|dokun)\s+(.+)$", part, re.I)
        if m:
            actions.append({"action": "click", "target": m.group(1).strip(" .'\"")})
            continue

        m = re.search(r"(?:type|yaz|gir)\s+[\"']([^\"']+)[\"']", part, re.I)
        if not m:
            m = re.search(r"^(.+?)\s+(?:yaz|gir|type)$", part, re.I)
        if m:
            value = m.group(1).strip(" .'\"")
            actions.append({"action": "type_text", "text": value})
            continue

        m = re.search(r"(?:press|bas)\s+(enter|return|back|home)", q)
        if not m:
            m = re.search(r"(enter|return|back|home)\s+(?:tusuna\s+)?bas", q)
        if m:
            key = m.group(1).upper().replace("RETURN", "ENTER")
            actions.append({"action": "press_key", "key": key})
            continue

        if "scroll" in q or "kaydir" in q:
            direction = "up" if ("up" in q or "yukari" in q) else "down"
            actions.append({"action": "scroll", "target": direction, "duration_ms": 450})
            continue

    if not actions:
        return None
    return {
        "goal": raw,
        "platform": "android",
        "actions": actions,
        "requires_visual_feedback": True,
        "requires_confirmation": True,
    }

def _general_answer(text):
    n = _norm(text)

    if re.search(r"\b(sen kimsin|kimsin|adin ne|adın ne)\b", text.lower()):
        return "Ben FLM. Bu Android sürümünde sembolik/semantic çekirdeğim ve Python runtime'ım doğrudan APK'nın içinde, tamamen yerel çalışıyor."

    if re.search(r"\b(nasilsin|nasılsın)\b", text.lower()):
        return "İyiyim; yerel FLM çekirdeği çalışıyor ve isteğini işlemeye hazırım."

    if "neler yapabilirsin" in n or "ne yapabilirsin" in n:
        return (
            "Yerel olarak matematik çözebilir, temel bilgi sorularını yanıtlayabilir, Python/JavaScript kodu üretebilir "
            "ve onayınla Android'de uygulama açma, dokunma, yazma, geri/home ve kaydırma eylemlerini planlayabilirim."
        )

    for key, answer in BASIC_FACTS.items():
        if key in n:
            return answer

    if "python" in n and "javascript" in n and ("karsilastir" in n or "fark" in n):
        return (
            "Python genel amaçlı ve veri/otomasyon tarafında güçlüdür; JavaScript web tarayıcılarının yerel dilidir ve "
            "Node.js ile sunucu tarafında da çalışır. Python genellikle girintiye, JavaScript ise süslü parantezlere dayanır."
        )

    return (
        "Bu konu için yerel semantic belleğimde yeterli kesin bilgi bulamadım. "
        "Uydurmak yerine bunu açıkça söylüyorum. İstersen soruyu daha somut bir hedefle yeniden sor."
    )

def initialize():
    global _turn
    with _lock:
        return json.dumps({
            "ok": True,
            "version": "2.6-android-embedded",
            "python": sys.version.split()[0],
            "embedded": True,
            "platform": "android",
            "neural_trainable_parameters": 0,
            "mobile_core": True,
        }, ensure_ascii=False)

def ask_json(text):
    global _turn, _last_answer
    text = (text or "").strip()
    if not text:
        return json.dumps({"ok": False, "error": "empty_prompt"}, ensure_ascii=False)

    with _lock:
        _turn += 1

        plan = _computer_plan(text)
        if plan is not None:
            answer = f"Telefon kontrol planı hazır: {len(plan['actions'])} adım. Çalıştırmadan önce senden onay isteyeceğim."
            _last_answer = answer
            return json.dumps({
                "ok": True,
                "kind": "computer",
                "answer": answer,
                "plan": plan,
                "turn": _turn,
            }, ensure_ascii=False)

        answer = _math_answer(text)
        if answer is None:
            answer = _code_answer(text)
        if answer is None:
            answer = _general_answer(text)

        _last_answer = answer
        return json.dumps({
            "ok": True,
            "kind": "chat",
            "answer": answer,
            "turn": _turn,
        }, ensure_ascii=False)

def health_json():
    return initialize()
