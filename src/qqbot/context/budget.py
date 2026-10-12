class ContextOverflowError(RuntimeError):
    """
    当前 Model Context 超出允许的输入 Token Budget。
    """

    def __init__(
        self,
        *,
        token_count: int,
        input_budget: int,
    ) -> None:
        self.token_count = token_count
        self.input_budget = input_budget

        super().__init__(f"上下文超出 Token Budget: {token_count}/{input_budget}")


class TokenBudgetManager:
    """
    管理模型上下文窗口。

    Context Window 包含：
        Input Tokens
        +
        Output Tokens

    因此：

        input_budget
        =
        context_window
        - max_output_tokens
        - safety_margin
    """

    def __init__(
        self,
        *,
        context_window: int,
        max_output_tokens: int,
        safety_margin: int,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window 必须大于 0")

        if max_output_tokens < 0:
            raise ValueError("max_output_tokens 不能小于 0")

        if safety_margin < 0:
            raise ValueError("safety_margin 不能小于 0")

        if max_output_tokens + safety_margin >= context_window:
            raise ValueError("max_output_tokens + safety_margin 不能占满 context_window")

        self.context_window = context_window

        self.max_output_tokens = max_output_tokens

        self.safety_margin = safety_margin

    @property
    def input_budget(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin

    def fits(
        self,
        token_count: int,
    ) -> bool:
        return token_count <= self.input_budget

    def ensure_fits(
        self,
        token_count: int,
    ) -> None:
        if self.fits(token_count):
            return

        raise ContextOverflowError(
            token_count=token_count,
            input_budget=self.input_budget,
        )

    def overflow_by(
        self,
        token_count: int,
    ) -> int:
        return max(
            0,
            token_count - self.input_budget,
        )

    def remaining(
        self,
        token_count: int,
    ) -> int:
        return max(
            0,
            self.input_budget - token_count,
        )
