from __future__ import annotations

import ast
import json
import math
import operator
from typing import Callable

from pydantic import Field

from .base import (
    BaseTool,
    ToolArgs,
)


class CalculatorArgs(ToolArgs):
    expression: str = Field(
        min_length=1,
        max_length=500,
        description=("数学表达式，例如 '(12 + 8) * 3 / 2'、'sqrt(144)'、'sin(pi / 2)'"),
    )


class CalculatorTool(BaseTool[CalculatorArgs]):
    name = "calculator"

    description = "执行精确的数学计算。遇到算术、百分比、幂、开方、三角函数、对数等计算时使用，不要依赖语言模型心算。"

    args_model = CalculatorArgs

    _binary_operators: dict[
        type[ast.operator],
        Callable,
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: (operator.floordiv),
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _unary_operators: dict[
        type[ast.unaryop],
        Callable,
    ] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _functions: dict[
        str,
        Callable,
    ] = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    _constants = {
        "pi": math.pi,
        "e": math.e,
    }

    async def run(
        self,
        args: CalculatorArgs,
    ) -> str:
        tree = ast.parse(
            args.expression,
            mode="eval",
        )

        nodes = list(ast.walk(tree))

        if len(nodes) > 100:
            raise ValueError("表达式过于复杂")

        result = self._eval(tree.body)

        if isinstance(
            result,
            float,
        ):
            if not math.isfinite(result):
                raise ValueError("计算结果不是有限数值")

        if (
            isinstance(
                result,
                (int, float),
            )
            and abs(result) > 1e100
        ):
            raise ValueError("计算结果过大")

        return json.dumps(
            {
                "expression": (args.expression),
                "result": result,
            },
            ensure_ascii=False,
        )

    def _eval(
        self,
        node: ast.AST,
    ) -> int | float:
        if isinstance(
            node,
            ast.Constant,
        ):
            value = node.value

            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError("只允许数字")

            return value

        if isinstance(
            node,
            ast.BinOp,
        ):
            operator_type = type(node.op)

            operation = self._binary_operators.get(operator_type)

            if operation is None:
                raise ValueError("不支持该运算符")

            left = self._eval(node.left)

            right = self._eval(node.right)

            if isinstance(
                node.op,
                ast.Pow,
            ):
                if abs(right) > 100:
                    raise ValueError("指数过大")

            return operation(
                left,
                right,
            )

        if isinstance(
            node,
            ast.UnaryOp,
        ):
            operation = self._unary_operators.get(type(node.op))

            if operation is None:
                raise ValueError("不支持该一元运算符")

            return operation(self._eval(node.operand))

        if isinstance(
            node,
            ast.Name,
        ):
            value = self._constants.get(node.id)

            if value is None:
                raise ValueError(f"未知常量: {node.id}")

            return value

        if isinstance(
            node,
            ast.Call,
        ):
            if not isinstance(
                node.func,
                ast.Name,
            ):
                raise ValueError("不允许该函数调用")

            function = self._functions.get(node.func.id)

            if function is None:
                raise ValueError(f"不支持函数: {node.func.id}")

            if node.keywords:
                raise ValueError("不支持关键字参数")

            values = [self._eval(arg) for arg in node.args]

            return function(*values)

        raise ValueError("表达式包含不允许的语法")
