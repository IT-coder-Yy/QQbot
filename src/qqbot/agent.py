from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .context import ApproxTokenCounter, ContextBuilder, ContextCompactor, ContextManager, TokenBudgetManager
from .schema import (
    AgentRun,
    ContextMessage,
    ContextRole,
    Conversation,
    MessageReceived,
    ModelContext,
    Session,
    SessionSummary,
    ToolCall as ToolCallRecord,
    ToolResult as ToolResultRecord,
)
from .tools import ToolRegistry


PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.md"

SESSION_COMPACT_PROMPT_PATH = PROMPTS_DIR / "session_compact.md"


class Agent:
    """
    QQ Bot 的核心 Agent。

    职责：

        Message
          ↓
        ContextBuilder
          ↓
        Token Budget
          ↓
        必要时 Session Compact
          ↓
        LLM
          ↓
        Tool Call?
        ├── YES
        │     ↓
        │   ToolRegistry
        │     ↓
        │   Tool Result
        │     ↓
        │   再次调用 LLM
        │
        └── NO
              ↓
          Final Answer

    Agent 不负责：

        - QQ WebSocket
        - Router
        - Intent Detection
        - Context Store 的具体存储方式
        - Tool 的具体实现
    """

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        request_timeout: float = 60.0,
    ) -> None:
        self.context_manager = context_manager
        self.context_builder = context_builder
        self.tool_registry = tool_registry

        # ====================================================
        # LLM Configuration
        # ====================================================

        self.api_key = self._get_required_env("LLM_API_KEY")
        self.base_url = self._get_required_env("LLM_BASE_URL")
        self.model = self._get_required_env("LLM_MODEL")

        # ====================================================
        # Model Context Limits
        # ====================================================

        self.context_window = self._get_required_int_env("LLM_CONTEXT_WINDOW")
        self.max_output_tokens = self._get_required_int_env("LLM_MAX_OUTPUT_TOKENS")
        self.safety_margin = self._get_required_int_env("LLM_CONTEXT_SAFETY_MARGIN")

        # ====================================================
        # Agent Tool Loop
        # ====================================================

        self.max_tool_steps = 6
        self.max_tool_result_tokens = 4000

        # ====================================================
        # Prompt
        # ====================================================

        self.system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

        # ====================================================
        # Context Harness
        # ====================================================

        self.token_counter = ApproxTokenCounter()
        self.token_budget = TokenBudgetManager(
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            safety_margin=self.safety_margin,
        )

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=request_timeout)

        # ====================================================
        # Context Compactor
        # ====================================================

        self.context_compactor = ContextCompactor(
            client=self.client,
            model=self.model,
            token_counter=self.token_counter,
            prompt_path=SESSION_COMPACT_PROMPT_PATH,
            max_output_tokens=self.max_output_tokens,
        )

    # ========================================================
    # Agent Entry
    # ========================================================

    async def run(
        self,
        *,
        message: MessageReceived,
        conversation: Conversation,
        session: Session,
        current_turn: ContextMessage,
    ) -> str:
        """
        执行完整的一轮 Agent Run。

        一轮 Run 可能经历：

            User
              ↓
            LLM
              ↓
            Tool Call
              ↓
            Tool Result
              ↓
            LLM
              ↓
            Tool Call
              ↓
            ...
              ↓
            Final Answer
        """

        run = AgentRun(
            run_id=self._new_run_id(message.message_id),
            conversation_id=conversation.conversation_id,
            session_id=session.session_id,
            current_turn=current_turn,
            started_at=current_turn.timestamp,
        )

        # ----------------------------------------------------
        # 当前这一轮 Agent 执行过程产生的消息。
        #
        # 注意：
        # 这些不是长期 Session History。
        #
        # 例如：
        #
        # assistant(tool_calls)
        # tool(result)
        # assistant(tool_calls)
        # tool(result)
        #
        # 只属于当前 Agent Run。
        # ----------------------------------------------------

        runtime_messages: list[ChatCompletionMessageParam] = []

        tool_round = 0

        while True:
            # =================================================
            # 1. 准备 Context
            # =================================================

            messages = await self._prepare_model_messages(
                conversation=(conversation),
                session=session,
                run=run,
                runtime_messages=(runtime_messages),
            )

            # =================================================
            # 2. 调用模型
            # =================================================

            response = await self._call_model(messages)

            self._print_model_usage(response)

            response_message = response.choices[0].message

            tool_calls = response_message.tool_calls or []

            # =================================================
            # 3. 没有 Tool Call
            #
            #    → Final Answer
            # =================================================

            if not tool_calls:
                content = response_message.content

                if not content:
                    return "模型没有返回内容。"

                content = content.strip()

                finished_at = datetime.now(tz=(current_turn.timestamp.tzinfo))

                run.final_response = content

                run.finished_at = finished_at

                # 由调用方在 OneBot 确认发送成功后写入会话历史。
                return content

            # =================================================
            # 4. Tool Loop 限制
            # =================================================

            tool_round += 1

            if tool_round > self.max_tool_steps:
                raise RuntimeError(f"Agent Tool Calling 超过最大轮数: {self.max_tool_steps}")

            # =================================================
            # 5. 保存模型产生的 assistant tool_call 消息
            # =================================================

            runtime_messages.append(self._assistant_tool_message(response_message))

            # =================================================
            # 6. 执行所有 Tool Calls
            # =================================================

            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                raw_arguments = tool_call.function.arguments

                print(f"[Tool Call] name={tool_name} arguments={raw_arguments}")

                parsed_arguments = self._parse_tool_arguments(raw_arguments)

                now = datetime.now(tz=(current_turn.timestamp.tzinfo))

                # =============================================
                # 6.1 记录领域层 ToolCall
                # =============================================

                run.tool_trace.append(
                    ToolCallRecord(
                        call_id=(tool_call.id),
                        tool_name=(tool_name),
                        arguments=(parsed_arguments),
                        timestamp=now,
                    )
                )

                # =============================================
                # 6.2 真正执行 Tool
                # =============================================

                raw_result = await self.tool_registry.execute(
                    name=tool_name,
                    arguments=(raw_arguments),
                )

                # =============================================
                # 6.3 Tool Result 长度保护
                # =============================================

                (
                    observation,
                    original_tokens,
                    current_tokens,
                    compacted,
                ) = self._limit_tool_result(raw_result)

                print(f"[Tool Result] name={tool_name} tokens≈{original_tokens} -> {current_tokens} compacted={compacted}")

                # =============================================
                # 6.4 保存领域层 ToolResult
                # =============================================

                run.tool_trace.append(
                    ToolResultRecord(
                        call_id=(tool_call.id),
                        tool_name=(tool_name),
                        observation=(observation),
                        raw_result_ref=None,
                        is_compacted=(compacted),
                        original_token_count=(original_tokens),
                        current_token_count=(current_tokens),
                        timestamp=(datetime.now(tz=(current_turn.timestamp.tzinfo))),
                    )
                )

                # =============================================
                # 6.5 Tool Result 返回模型
                # =============================================

                runtime_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (tool_call.id),
                        "content": (observation),
                    }
                )

            # -------------------------------------------------
            # 当前 tool_calls 全部执行完。
            #
            # 回 while 顶部：
            #
            # Context Build
            # → Token Budget
            # → Model
            #
            # -------------------------------------------------

    # ========================================================
    # Model Request
    # ========================================================

    async def _call_model(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> Any:
        """
        真正调用主模型。

        ToolRegistry 中注册的工具全部暴露给模型。
        """

        tool_definitions = self.tool_registry.definitions

        request: dict[
            str,
            Any,
        ] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": (self.max_output_tokens),
        }

        # 如果一个工具都没有，
        # 就不要传 tools=[]，
        # 兼容更多 OpenAI-compatible Provider。
        if tool_definitions:
            request["tools"] = tool_definitions

            request["tool_choice"] = "auto"

        return await self.client.chat.completions.create(**request)

    # ========================================================
    # Context Preparation
    # ========================================================

    async def _prepare_model_messages(
        self,
        *,
        conversation: Conversation,
        session: Session,
        run: AgentRun,
        runtime_messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """
        Context Harness 核心循环。

        Build
          ↓
        Render
          ↓
        Count
          ↓
        Fits?
        │
        ├── YES
        │     ↓
        │   返回 messages
        │
        └── NO
              ↓
          Session Compact
              ↓
          再 Build

        注意：

        只有真正超过 Input Budget
        才进行压缩。

        不使用 70%、80% 之类的提前压缩策略。
        """

        while True:
            # =================================================
            # 1. 构建 Domain ModelContext
            # =================================================

            model_context = self.context_builder.build(
                system_prompt=(self.system_prompt),
                conversation=(conversation),
                session=session,
                run=run,
            )

            # =================================================
            # 2. 转换成 LLM Messages
            # =================================================

            base_messages = self._build_model_messages(model_context)

            messages = [
                *base_messages,
                *runtime_messages,
            ]

            # =================================================
            # 3. Tool Definitions 也占 Context
            # =================================================

            tool_schema_tokens = self._count_tool_schema_tokens()

            # =================================================
            # 4. 计算 Context Token
            # =================================================

            estimated_tokens = self.token_counter.count_messages(messages) + tool_schema_tokens

            self._print_context_usage(estimated_tokens)

            # =================================================
            # 5. 可以放下
            # =================================================

            if self.token_budget.fits(estimated_tokens):
                return messages

            # =================================================
            # 6. Overflow
            # =================================================

            overflow_tokens = self.token_budget.overflow_by(estimated_tokens)

            print(f"[Context Overflow] overflow≈{overflow_tokens}")

            before_count = len(session.recent_messages)

            # =================================================
            # 7. 压缩最老的 Session Messages
            # =================================================

            compacted = await self.context_compactor.compact_session(
                session=session,
                current_turn_id=(run.current_turn.id),
                overflow_tokens=(overflow_tokens),
                max_source_tokens=(self.token_budget.input_budget // 2),
            )

            # =================================================
            # 8. 已经没有东西可以继续压缩
            # =================================================

            if not compacted:
                self.token_budget.ensure_fits(estimated_tokens)

            after_count = len(session.recent_messages)

            # =================================================
            # 9. 防止死循环
            # =================================================

            if after_count >= before_count:
                raise RuntimeError("Context Compactor 没有减少 Recent Messages，停止继续压缩")

            # =================================================
            # 10. 保存 Compact 后 Session
            # =================================================

            self.context_manager.save_session(session)

            print(f"[Context Compact] recent_messages={before_count} -> {after_count}")

            # -------------------------------------------------
            # 不直接认为压缩后一定够。
            #
            # 回 while 顶部重新：
            #
            # Build
            # Count
            # Fits
            #
            # -------------------------------------------------

    # ========================================================
    # Model Context Renderer
    # ========================================================

    def _build_model_messages(
        self,
        context: ModelContext,
    ) -> list[ChatCompletionMessageParam]:
        """
        Domain ModelContext
        ↓
        OpenAI / DeepSeek messages
        """

        messages: list[ChatCompletionMessageParam] = []

        # ====================================================
        # System
        # ====================================================

        system_content = self._build_system_context(context)

        messages.append(
            {
                "role": "system",
                "content": (system_content),
            }
        )

        # ====================================================
        # Recent Messages
        # ====================================================

        for context_message in context.recent_messages:
            messages.append(self._context_message_to_llm(context_message))

        # ====================================================
        # Current Turn
        # ====================================================

        messages.append(self._context_message_to_llm(context.current_turn))

        return messages

    def _build_system_context(
        self,
        context: ModelContext,
    ) -> str:
        """
        当前阶段使用：

            System Prompt
            +
            Session Summary

        Long-term Memory 暂时不启用。

        原因：
            当前 Store 还是 InMemory，
            暂时不存在真正的长期持久化记忆。
        """

        parts = [
            context.system_prompt,
        ]

        summary_text = self._render_summary(context.session_summary)

        if summary_text:
            parts.append("## 当前 Session 摘要\n" + summary_text)

        return "\n\n".join(parts)

    def _context_message_to_llm(
        self,
        message: ContextMessage,
    ) -> ChatCompletionMessageParam:
        """
        ContextMessage
        ↓
        ChatCompletionMessageParam
        """

        # ====================================================
        # Assistant
        # ====================================================

        if message.role is ContextRole.ASSISTANT:
            return {
                "role": "assistant",
                "content": (message.content),
            }

        # ====================================================
        # User
        # ====================================================

        content = message.content

        # QQ 群环境需要告诉模型：
        #
        # 谁说了这句话。
        #
        # 否则多个人都会变成普通 role=user，
        # 模型无法区分说话者。
        if message.sender_name:
            content = f"[{message.sender_name}] {content}"

        return {
            "role": "user",
            "content": content,
        }

    # ========================================================
    # Assistant Tool Call Renderer
    # ========================================================

    def _assistant_tool_message(
        self,
        message: Any,
    ) -> ChatCompletionMessageParam:
        """
        把模型产生的：

            assistant
            +
            tool_calls

        重新放回下一轮 Context。

        DeepSeek Thinking Mode 下，
        reasoning_content 也需要一起保留。
        """

        tool_calls = []

        for tool_call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": (tool_call.id),
                    "type": ("function"),
                    "function": {
                        "name": (tool_call.function.name),
                        "arguments": (tool_call.function.arguments),
                    },
                }
            )

        payload: dict[
            str,
            Any,
        ] = {
            "role": "assistant",
            "content": (message.content),
            "tool_calls": (tool_calls),
        }

        # ----------------------------------------------------
        # DeepSeek reasoning model
        #
        # 如果返回 reasoning_content，
        # 后续 Tool Calling 请求需要继续带回。
        # ----------------------------------------------------

        reasoning_content = getattr(
            message,
            "reasoning_content",
            None,
        )

        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content

        return cast(
            ChatCompletionMessageParam,
            payload,
        )

    # ========================================================
    # Session Summary Renderer
    # ========================================================

    @staticmethod
    def _render_summary(
        summary: SessionSummary,
    ) -> str:
        sections: list[str] = []

        if summary.narrative:
            sections.append(summary.narrative)

        if summary.goals:
            sections.append("目标：" + "；".join(summary.goals))

        if summary.facts:
            sections.append("事实：" + "；".join(summary.facts))

        if summary.decisions:
            sections.append("已确认决策：" + "；".join(summary.decisions))

        if summary.unresolved_questions:
            sections.append("待解决问题：" + "；".join(summary.unresolved_questions))

        if summary.important_entities:
            sections.append("重要实体：" + "；".join(summary.important_entities))

        return "\n".join(sections)

    # ========================================================
    # Tool Result Protection
    # ========================================================

    def _limit_tool_result(
        self,
        result: str,
    ) -> tuple[
        str,
        int,
        int,
        bool,
    ]:
        """
        Tool Result 是 Context 中最容易爆炸的部分。

        当前阶段采用：

            Hard Token Limit
            +
            截断

        后面可以升级为真正的：

            ToolCompactor
              ↓
            LLM Summary
        """

        original_tokens = self.token_counter.count_text(result)

        # ====================================================
        # 没超过限制
        # ====================================================

        if original_tokens <= self.max_tool_result_tokens:
            return (
                result,
                original_tokens,
                original_tokens,
                False,
            )

        # ====================================================
        # 超过限制
        # ====================================================

        suffix = "\n\n[Tool Result 已截断：原始工具结果过长]"

        # ----------------------------------------------------
        # 用二分查找找到：
        #
        # 最大可保留字符长度
        #
        # 而不是简单按照字符比例截断。
        # ----------------------------------------------------

        low = 0

        high = len(result)

        while low < high:
            middle = (low + high + 1) // 2

            candidate = result[:middle] + suffix

            candidate_tokens = self.token_counter.count_text(candidate)

            if candidate_tokens <= self.max_tool_result_tokens:
                low = middle

            else:
                high = middle - 1

        observation = result[:low] + suffix

        current_tokens = self.token_counter.count_text(observation)

        return (
            observation,
            original_tokens,
            current_tokens,
            True,
        )

    # ========================================================
    # Tool Schema Token Count
    # ========================================================

    def _count_tool_schema_tokens(
        self,
    ) -> int:
        """
        Tool definitions 本身也会被发送给模型，
        因此必须计入 Context Budget。
        """

        definitions = self.tool_registry.definitions

        if not definitions:
            return 0

        serialized = json.dumps(
            definitions,
            ensure_ascii=False,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )

        return self.token_counter.count_text(serialized)

    # ========================================================
    # Tool Arguments
    # ========================================================

    @staticmethod
    def _parse_tool_arguments(
        raw_arguments: str,
    ) -> dict[str, Any]:
        """
        这里只是为了把 ToolCall 记录进 AgentRun。

        真正的参数验证仍然由：

            BaseTool
            +
            Pydantic args_model

        完成。
        """

        try:
            result = json.loads(raw_arguments)

        except json.JSONDecodeError:
            return {"_raw": (raw_arguments)}

        if not isinstance(
            result,
            dict,
        ):
            return {
                "_raw": result,
            }

        return result

    # ========================================================
    # Observability
    # ========================================================

    def _print_context_usage(
        self,
        estimated_tokens: int,
    ) -> None:
        remaining = self.token_budget.remaining(estimated_tokens)

        print(f"[Context] input≈{estimated_tokens} budget={self.token_budget.input_budget} remaining≈{remaining} window={self.context_window}")

    @staticmethod
    def _print_model_usage(
        response: Any,
    ) -> None:
        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return

        print(f"[LLM Usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _new_run_id(
        message_id: int,
    ) -> str:
        return f"run:{message_id}"

    @staticmethod
    def _get_required_env(
        name: str,
    ) -> str:
        value = os.getenv(name)

        if not value:
            raise RuntimeError(f"缺少环境变量: {name}")

        return value

    @staticmethod
    def _get_required_int_env(
        name: str,
    ) -> int:
        value = os.getenv(name)

        if not value:
            raise RuntimeError(f"缺少环境变量: {name}")

        try:
            return int(value)

        except ValueError as exc:
            raise RuntimeError(f"环境变量 {name} 必须是整数") from exc
