from __future__ import annotations

import ast
import math
import operator


_BIN={
    ast.Add:operator.add,
    ast.Sub:operator.sub,
    ast.Mult:operator.mul,
    ast.Div:operator.truediv,
    ast.FloorDiv:operator.floordiv,
    ast.Mod:operator.mod,
    ast.Pow:operator.pow,
}
_UN={ast.UAdd:operator.pos,ast.USub:operator.neg}


class ArithmeticSolver:
    """Safe deterministic arithmetic evaluator; no eval() and no model guessing."""

    def __init__(self,*,max_nodes:int=128,max_abs:float=1e100):
        self.max_nodes=max_nodes
        self.max_abs=max_abs

    def solve(self,expression:str):
        tree=ast.parse(expression,mode="eval")
        nodes=list(ast.walk(tree))
        if len(nodes)>self.max_nodes:
            raise ValueError("expression too complex")
        value=self._node(tree.body)
        if isinstance(value,(int,float)) and abs(value)>self.max_abs:
            raise OverflowError("result exceeds solver bound")
        return value

    def _node(self,node):
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):
            return node.value
        if isinstance(node,ast.BinOp) and type(node.op) in _BIN:
            a=self._node(node.left); b=self._node(node.right)
            if isinstance(node.op,ast.Pow) and abs(b)>12:
                raise ValueError("power exponent too large")
            return _BIN[type(node.op)](a,b)
        if isinstance(node,ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](self._node(node.operand))
        raise ValueError(f"unsupported arithmetic syntax: {type(node).__name__}")
