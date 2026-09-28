from __future__ import annotations

from flm.core.runtime import FLMCore
from flm.core.semantic_compiler import SemanticCompiler
from flm.core.renderer import DeterministicRenderer


class TrainingFreeAgent:
    """No-checkpoint FLM path for canonical language/code/core operations."""

    def __init__(self, core: FLMCore | None = None):
        self.core=core or FLMCore()
        self.compiler=SemanticCompiler()
        self.renderer=DeterministicRenderer()

    def ask(self, prompt: str) -> str:
        program=self.compiler.compile_any(prompt)
        response=self.core.execute(program)
        return self.renderer.render(prompt,response)
