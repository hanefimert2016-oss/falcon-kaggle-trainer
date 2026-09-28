from __future__ import annotations

from dataclasses import dataclass,field
import ast
from pathlib import Path
import re


@dataclass
class CodeSymbol:
    name:str
    kind:str
    line:int


@dataclass
class CodeGraph:
    language:str
    symbols:list[CodeSymbol]=field(default_factory=list)
    calls:set[tuple[str,str]]=field(default_factory=set)
    imports:set[str]=field(default_factory=set)
    diagnostics:list[str]=field(default_factory=list)


class CodeEngine:
    """Training-free structural code analyzer used by FLM Core."""

    def analyze_python(self,source:str)->CodeGraph:
        graph=CodeGraph(language="python")
        try:
            tree=ast.parse(source)
        except SyntaxError as exc:
            graph.diagnostics.append(
                f"SyntaxError line={exc.lineno} offset={exc.offset}: {exc.msg}"
            )
            return graph

        scope=["<module>"]
        class Visitor(ast.NodeVisitor):
            def visit_Import(self,node):
                for alias in node.names:
                    graph.imports.add(alias.name)
            def visit_ImportFrom(self,node):
                if node.module:
                    graph.imports.add(node.module)
            def visit_ClassDef(self,node):
                graph.symbols.append(CodeSymbol(node.name,"class",node.lineno))
                scope.append(node.name); self.generic_visit(node); scope.pop()
            def visit_FunctionDef(self,node):
                graph.symbols.append(CodeSymbol(node.name,"function",node.lineno))
                scope.append(node.name); self.generic_visit(node); scope.pop()
            visit_AsyncFunctionDef=visit_FunctionDef
            def visit_Call(self,node):
                target=None
                if isinstance(node.func,ast.Name):
                    target=node.func.id
                elif isinstance(node.func,ast.Attribute):
                    target=node.func.attr
                if target:
                    graph.calls.add((scope[-1],target))
                self.generic_visit(node)
        Visitor().visit(tree)
        return graph

    def analyze_text(self,source:str,language:str)->CodeGraph:
        lang=language.lower()
        if lang in {"py","python"}:
            return self.analyze_python(source)
        graph=CodeGraph(language=lang)
        # Deterministic fallback symbol extraction for C-like languages.
        pattern=re.compile(
            r"(?m)^\s*(?:public|private|protected|static|final|async|export|func|fn|def|class|struct|interface|void|int|string|bool|[A-Z][\w<>]*)"
            r"(?:\s+[\w<>\[\],?]+)*\s+([A-Za-z_]\w*)\s*(?:\(|\{)"
        )
        for i,line in enumerate(source.splitlines(),1):
            m=pattern.search(line)
            if m:
                graph.symbols.append(CodeSymbol(m.group(1),"symbol",i))
        return graph
