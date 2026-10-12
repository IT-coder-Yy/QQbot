import asyncio
import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import qqbot
from qqbot.agent import Agent
from qqbot.context import ContextBuilder, ContextManager, InMemoryContextStore
from qqbot.listen import main
from qqbot.settings import Settings
from qqbot.tools import ToolRegistry, build_default_tool_registry
from qqbot.tools.calculator import CalculatorTool
from tests.helpers import message


class SettingsTests(unittest.TestCase):
    def test_connection_config_uses_code_runtime_settings(self):
        env = {
            "NAPCAT_TOKEN": "test",
            "NAPCAT_WS_URL": "wss://example.invalid/onebot",
            "BOT_CONCURRENCY": "0",
            "BOT_TASK_TIMEOUT": "nan",
            "NAPCAT_ACTION_TIMEOUT": "inf",
            "JEV_REPLY_THRESHOLD": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings, Settings(ws_url=env["NAPCAT_WS_URL"], token="test"))

    def test_invalid_settings_fail_before_connecting(self):
        cases = [
            {"NAPCAT_TOKEN": ""},
            {"NAPCAT_TOKEN": "  "},
            {"NAPCAT_WS_URL": "https://example.invalid"},
        ]
        for case in cases:
            with self.subTest(case=case), patch.dict(os.environ, {"NAPCAT_TOKEN": "test", **case}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_tavily_key_enables_search(self):
        with patch.dict(os.environ, {"TAVILY_SEARCH_API_KEY": "test-tavily"}, clear=True):
            registry = build_default_tool_registry()
        self.assertIn("web_search", registry.names)
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("web_search", build_default_tool_registry().names)

    def test_console_entry_runs_listener(self):
        with patch("qqbot.listen.main", new_callable=AsyncMock) as listener:
            qqbot.main()
        listener.assert_awaited_once()


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_jev_key_does_not_prevent_startup_and_resources_close(self):
        fake_agent = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
        captured = []
        cleanup_tasks = []

        async def serve_once(settings, transport, dispatcher):
            captured.extend([transport, dispatcher])
            cleanup_tasks.extend(task for task in asyncio.all_tasks() if task.get_name() == "context-cleanup")

        with patch.dict(os.environ, {"NAPCAT_TOKEN": "test"}, clear=True), patch("qqbot.listen.load_dotenv"), patch("qqbot.listen.logging.basicConfig"), patch("qqbot.listen.Agent", return_value=fake_agent), patch("qqbot.listen.AsyncTypeSafeClient") as jev, patch("qqbot.listen.serve", side_effect=serve_once):
            await main()
        jev.assert_not_called()
        fake_agent.client.close.assert_awaited_once()
        self.assertTrue(captured[0]._closed)
        self.assertTrue(captured[1]._closed)
        self.assertEqual(len(cleanup_tasks), 1)
        self.assertTrue(cleanup_tasks[0].cancelled())

    async def test_cancelling_service_cleans_all_resources(self):
        fake_agent = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
        ready = asyncio.Event()
        captured = []
        cleanup_tasks = []

        async def serve_forever(settings, transport, dispatcher):
            captured.extend([transport, dispatcher])
            cleanup_tasks.extend(task for task in asyncio.all_tasks() if task.get_name() == "context-cleanup")
            ready.set()
            await asyncio.Event().wait()

        with patch.dict(os.environ, {"NAPCAT_TOKEN": "test"}, clear=True), patch("qqbot.listen.load_dotenv"), patch("qqbot.listen.logging.basicConfig"), patch("qqbot.listen.Agent", return_value=fake_agent), patch("qqbot.listen.serve", side_effect=serve_forever):
            task = asyncio.create_task(main())
            await ready.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        fake_agent.client.close.assert_awaited_once()
        self.assertTrue(captured[0]._closed)
        self.assertTrue(captured[1]._closed)
        self.assertEqual(len(cleanup_tasks), 1)
        self.assertTrue(cleanup_tasks[0].cancelled())

    async def test_real_agent_tool_loop_returns_answer_without_recording_unsent_reply(self):
        def response(content=None, calls=None):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or [], reasoning_content=None))])

        tool_call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="calculator", arguments='{"expression":"2+3"}'))
        completion = AsyncMock(side_effect=[response(calls=[tool_call]), response(content="答案是 5")])
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        env = {"LLM_API_KEY": "test", "LLM_BASE_URL": "https://example.invalid", "LLM_MODEL": "test", "LLM_CONTEXT_WINDOW": "16000", "LLM_MAX_OUTPUT_TOKENS": "1000", "LLM_CONTEXT_SAFETY_MARGIN": "1000"}
        env.update(AGENT_MAX_TOOL_STEPS="0", TOOL_MAX_RESULT_TOKENS="invalid")
        manager = ContextManager(InMemoryContextStore())
        with patch.dict(os.environ, env, clear=True), patch("qqbot.agent.AsyncOpenAI", return_value=fake_client):
            agent = Agent(context_manager=manager, context_builder=ContextBuilder(), tool_registry=ToolRegistry([CalculatorTool()]))
        incoming = message()
        conversation, session, turn = manager.record_user_message(incoming)
        with contextlib.redirect_stdout(io.StringIO()):
            answer = await agent.run(message=incoming, conversation=conversation, session=session, current_turn=turn)
        self.assertEqual(answer, "答案是 5")
        self.assertEqual(completion.await_count, 2)
        self.assertEqual(len(session.recent_messages), 1)
        self.assertEqual(completion.call_args.kwargs["messages"][-1]["role"], "tool")
