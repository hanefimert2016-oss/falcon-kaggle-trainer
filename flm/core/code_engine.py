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
        aliases={
            "py":"python","js":"javascript","ts":"typescript","c++":"cpp",
            "cxx":"cpp","rs":"rust","golang":"go",
        }
        lang=aliases.get(language.lower().strip(),language.lower().strip())
        if lang=="python":
            return self.analyze_python(source)

        graph=CodeGraph(language=lang)
        lines=source.splitlines()

        import_patterns={
            "java":[r"^\s*import\s+([\w.]+)\s*;"],
            "javascript":[
                r"^\s*import(?:.+?from\s+)?[\"']([^\"']+)[\"']",
                r"require\(\s*[\"']([^\"']+)[\"']\s*\)",
            ],
            "typescript":[
                r"^\s*import(?:.+?from\s+)?[\"']([^\"']+)[\"']",
                r"require\(\s*[\"']([^\"']+)[\"']\s*\)",
            ],
            "go":[r"^\s*import\s+[\"']([^\"']+)[\"']"],
            "rust":[r"^\s*use\s+([^;]+)\s*;"],
            "c":[r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]"],
            "cpp":[r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]"],
        }
        for line in lines:
            for pat in import_patterns.get(lang,[]):
                m=re.search(pat,line)
                if m:
                    graph.imports.add(m.group(1).strip())

        class_pat=re.compile(
            r"\b(class|struct|interface|enum)\s+([A-Za-z_]\w*)"
        )
        function_patterns=[
            re.compile(r"\b(?:function|func|fn)\s+([A-Za-z_]\w*)\s*\("),
            re.compile(
                r"^\s*(?:(?:public|private|protected|static|final|virtual|inline|"
                r"constexpr|async|export|extern|synchronized)\s+)*"
                r"(?:[A-Za-z_]\w*(?:\s*<[^;{}()]+>)?(?:\[\])?[&*]?\s+)+"
                r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:\{|throws\b|$)"
            ),
            re.compile(
                r"^\s*(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*"
                r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>"
            ),
        ]

        function_lines:dict[int,str]={}
        for lineno,line in enumerate(lines,1):
            cm=class_pat.search(line)
            if cm:
                graph.symbols.append(CodeSymbol(cm.group(2),cm.group(1),lineno))
            for pat in function_patterns:
                m=pat.search(line)
                if m:
                    name=m.group(1)
                    if name not in {"if","for","while","switch","catch"}:
                        graph.symbols.append(CodeSymbol(name,"function",lineno))
                        function_lines[lineno]=name
                        break

        call_pat=re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
        keywords={
            "if","for","while","switch","catch","return","sizeof","typeof",
            "new","throw","synchronized","function","func","fn",
        }
        current_scope="<module>"
        for lineno,line in enumerate(lines,1):
            if lineno in function_lines:
                current_scope=function_lines[lineno]
            declared=function_lines.get(lineno)
            for match in call_pat.finditer(line):
                raw=match.group(1)
                target=raw.rsplit(".",1)[-1]
                if target in keywords or (declared and target==declared):
                    continue
                graph.calls.add((current_scope,target))

        if lang in {"java","javascript","typescript","go","rust","c","cpp"}:
            # Cheap structural diagnostic only; it deliberately does not pretend
            # to replace a real compiler.
            delta=source.count("{")-source.count("}")
            if delta:
                graph.diagnostics.append(f"UnbalancedBraces delta={delta}")

        # Preserve deterministic order and avoid duplicate symbol rows.
        unique={}
        for sym in graph.symbols:
            unique[(sym.name,sym.kind,sym.line)]=sym
        graph.symbols=list(unique.values())
        return graph
