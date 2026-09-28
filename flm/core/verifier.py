from __future__ import annotations

from dataclasses import dataclass
from typing import Callable,Any


@dataclass
class Verification:
    ok:bool
    checks:list[dict]


class Verifier:
    """Explicit post-condition verifier for Core plans and tool results."""

    def verify(self,checks:list[tuple[str,Callable[[],Any]]])->Verification:
        results=[]
        ok=True
        for name,fn in checks:
            try:
                value=fn()
                passed=bool(value)
                results.append({"name":name,"ok":passed,"value":value})
                ok=ok and passed
            except Exception as exc:
                results.append({
                    "name":name,"ok":False,
                    "error":type(exc).__name__,
                    "message":str(exc),
                })
                ok=False
        return Verification(ok,results)
