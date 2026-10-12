from __future__ import annotations

import json
import math
import re
from typing import (
    Any,
    Protocol,
    Sequence,
)


class TokenCounter(Protocol):
    def count_text(
        self,
        text: str,
    ) -> int: ...

    def count_messages(
        self,
        messages: Sequence[Any],
    ) -> int: ...


class ApproxTokenCounter:
    """
    通用 Token 估算器。

    不是模型的精确 tokenizer。
    实际 Token 以 API usage 为准。
    """

    _cjk_pattern = re.compile(
        r"[\u3400-\u4dbf"
        r"\u4e00-\u9fff"
        r"\uf900-\ufaff]"
    )

    def count_text(
        self,
        text: str,
    ) -> int:
        if not text:
            return 0

        cjk_count = len(self._cjk_pattern.findall(text))

        other_count = len(text) - cjk_count

        cjk_tokens = cjk_count

        other_tokens = math.ceil(other_count / 4)

        return max(
            1,
            cjk_tokens + other_tokens,
        )

    def count_messages(
        self,
        messages: Sequence[Any],
    ) -> int:
        total = 0

        for message in messages:
            serialized = json.dumps(
                message,
                ensure_ascii=False,
                default=str,
                separators=(
                    ",",
                    ":",
                ),
            )

            total += self.count_text(serialized)

            # Chat template
            # 粗略额外开销
            total += 4

        total += 2

        return total
