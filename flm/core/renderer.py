from __future__ import annotations

from .runtime import CoreResponse


def _is_turkish(text: str) -> bool:
    low=str(text or "").casefold()
    return any(x in low for x in (
        " mı"," mi"," mu"," mü"," bir ","hesapla","kodu","kodunu","analiz","sıcakkanlı",
        "kaydet","dosya","nedir","kaç","işlemi","memeli","bilgisayar","hayvan","araç",
    )) or any(ch in low for ch in "çğıöşü")


class DeterministicRenderer:
    """Training-free renderer for simple verified FLM Core results."""

    def render(self, prompt: str, response: CoreResponse) -> str:
        tr=_is_turkish(prompt)

        if response.results:
            item=response.results[0]
            return ("Evet." if item.answer else "Hayır.") if tr else ("Yes." if item.answer else "No.")

        if response.operation_results:
            item=response.operation_results[0]
            kind=item.get("kind")
            if not item.get("ok",False):
                msg=item.get("message") or item.get("error") or "unknown error"
                return (f"İşlem başarısız: {msg}" if tr else f"Operation failed: {msg}")

            if kind=="ARITHMETIC":
                return (f"Sonuç: {item.get('value')}." if tr else f"Result: {item.get('value')}.")

            if kind=="ANALYZE_CODE":
                diagnostics=item.get("diagnostics") or []
                if diagnostics:
                    first=diagnostics[0]
                    return (f"Kod analizi bir hata buldu: {first}" if tr else f"Code analysis found an error: {first}")
                symbols=[x.get("name","") for x in item.get("symbols") or [] if x.get("name")]
                calls=[" -> ".join(x) for x in item.get("calls") or []]
                if tr:
                    return f"Kod analizi tamamlandı. Semboller: {', '.join(symbols) or 'yok'}. Çağrılar: {', '.join(calls) or 'yok'}."
                return f"Code analysis complete. Symbols: {', '.join(symbols) or 'none'}. Calls: {', '.join(calls) or 'none'}."

            if kind=="STATE_PLAN":
                actions=item.get("actions") or []
                return (f"Plan: {' -> '.join(actions)}." if tr else f"Plan: {' -> '.join(actions)}.")

            if kind=="UI_PLAN":
                actions=item.get("actions") or []
                rendered=[str(x.get("op"))+(f"({x.get('target_id')})" if x.get("target_id") else "") for x in actions]
                return (f"Eylemler: {' -> '.join(rendered)}." if tr else f"Actions: {' -> '.join(rendered)}.")

            if kind=="VERIFY":
                return ("Tüm kontroller geçti." if tr else "All checks passed.")

        if response.memory_facts or response.memory_rules:
            return "Bilgi kaydedildi." if tr else "Knowledge recorded."

        return "Sonuç yok." if tr else "No result."
