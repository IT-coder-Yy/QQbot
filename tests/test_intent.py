import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from qqbot.chat_context import build_intent_state, direct_reply_decision, message_metadata
from qqbot.agent import Agent
from qqbot.context import ContextManager, InMemoryContextStore
from qqbot.context.sqlite_store import SQLiteContextStore
from qqbot.dispatcher import MessageJob
from qqbot.intent import IntentDetector, should_respond
from qqbot.listen import handle_message
from qqbot.schema import ContextAttachment, ContextRole, MentionContent, ReplyContent, TextContent
from scripts.evaluate_intent import make_case
from tests.helpers import message


class IntentStateTests(unittest.TestCase):
    def test_cold_start_and_identity_are_explicit(self):
        session, turn, incoming = make_case({'current':'你知道刚才发生了什么嘛？'})
        state = build_intent_state(session=session,current_turn=turn,message=incoming)
        self.assertEqual(state['recent_messages'], [])
        self.assertEqual(state['bot'], {'id':10,'names':['糖糖','村头洋哥']})
        self.assertEqual(state['current_message']['text'], incoming.text)
        self.assertIn('村头洋哥',state['context'])

    def test_mentions_quote_and_same_name_senders_remain_distinct(self):
        session, turn, incoming = make_case({'current':'那你推荐一下','history':[['user',21,'我知道一个地方'],['assistant',10,'想去哪玩？']],'reply_to':2,'mentions':[21],'same_names':True})
        state = build_intent_state(session=session,current_turn=turn,message=incoming)
        self.assertEqual(state['current_message']['mentioned_ids'],[21])
        self.assertEqual(state['quoted_message']['role'],'assistant')
        self.assertNotEqual(state['current_message']['sender_id'],state['recent_messages'][0]['sender_id'])
        incoming.content=[ReplyContent(message_id=1),TextContent(text=incoming.text)]
        other=build_intent_state(session=session,current_turn=turn,message=incoming)
        self.assertEqual(other['quoted_message']['role'],'user')
        self.assertNotEqual(state,other)

    def test_relevant_bot_turn_survives_group_chatter_without_duplicate_current(self):
        rows=[['user',20,'想出去玩'],['assistant',10,'去公园吗？']]+[['user',21,f'闲聊{i}'] for i in range(20)]
        session,turn,incoming=make_case({'current':'换一个吧','history':rows,'gap_seconds':3600})
        state=build_intent_state(session=session,current_turn=turn,message=incoming)
        ids=[row['message_id'] for row in state['recent_messages']]
        self.assertIn(1,ids)
        self.assertIn(2,ids)
        self.assertNotIn(1000,ids)
        self.assertLessEqual(len(ids),16)
        self.assertGreaterEqual(state['recent_messages'][0]['seconds_ago'],3600)

    def test_long_message_is_marked_truncated_for_judgment_only(self):
        session,turn,incoming=make_case({'current':'字'*10000})
        state=build_intent_state(session=session,current_turn=turn,message=incoming)
        self.assertEqual(len(state['current_message']['text']),1000)
        self.assertTrue(state['current_message']['text_truncated'])
        self.assertEqual(len(turn.content),10000)

    def test_picture_is_visible_to_intent_in_current_and_followup_turns(self):
        session, turn, incoming = make_case({'current': '', 'history': [['assistant', 10, '发来看看']]})
        turn.attachments = [ContextAttachment(type='image', id='picture', name='图片', url='https://example.invalid/private')]
        state = build_intent_state(session=session, current_turn=turn, message=incoming)
        expected = [{'type': 'image', 'id': 'picture', 'name': '图片'}]
        self.assertEqual(state['current_message']['attachments'], expected)
        followup = turn.model_copy(update={'id': 'followup', 'source_message_id': 1001,
                                           'content': '这图片有什么', 'attachments': []})
        state = build_intent_state(session=session, current_turn=followup)
        self.assertEqual(state['recent_messages'][-1]['attachments'], expected)
        self.assertEqual(state['current_message']['attachments'], [])

    def test_explicit_other_recipient_overrides_quote_bot(self):
        session,turn,incoming=make_case({'current':'你看到了吗','history':[['assistant',10,'周六集合']],'reply_to':1,'mentions':[21]})
        self.assertIsNone(direct_reply_decision(incoming,session))
        incoming.content.append(MentionContent(target_id=10))
        self.assertTrue(direct_reply_decision(incoming,session))

    def test_bot_name_with_other_mention_and_all_members_need_semantics(self):
        session,turn,incoming=make_case({'current':'糖糖也说一下','mentions':[21]})
        self.assertIsNone(direct_reply_decision(incoming,session))
        incoming.content=[TextContent(text='大家给点意见'),MentionContent(target_id=21),MentionContent(is_all=True)]
        self.assertIsNone(direct_reply_decision(incoming,session))

    def test_manager_preserves_addressing_metadata(self):
        manager=ContextManager(InMemoryContextStore())
        incoming=message(1,group_id=30)
        incoming.content.extend([MentionContent(target_id=21),ReplyContent(message_id=99)])
        _,_,turn=manager.record_user_message(incoming)
        self.assertEqual(turn.metadata['mentioned_ids'],[21])
        self.assertEqual(turn.metadata['reply_to_message_id'],99)
        self.assertEqual(turn.metadata['bot_id'],10)

    def test_main_model_gets_group_identity_and_resolvable_quote(self):
        session,turn,incoming=make_case({'current':'那怎么做','history':[['user',21,'我也想学'],['assistant',10,'先装软件']],'reply_to':2,'mentions':[10],'same_names':True})
        agent=Agent.__new__(Agent)
        agent.context_manager=ContextManager(InMemoryContextStore())
        agent.tool_registry=SimpleNamespace(names=[])
        agent.media=None
        context=SimpleNamespace(system_prompt='原有人设',current_turn=turn,session_summary=session.summary)
        system=agent._build_system_context(context)
        self.assertIn('QQ 多人群聊',system)
        self.assertIn('QQ 号是 10',system)
        self.assertIn('原有人设',system)
        current=agent._context_message_to_llm(turn)['content']
        previous=agent._context_message_to_llm(session.recent_messages[0])['content']
        quoted=agent._context_message_to_llm(session.recent_messages[1])['content']
        self.assertIn('QQ:20',current)
        self.assertIn('QQ:21',previous)
        self.assertIn('引用消息ID=2',current)
        self.assertIn('消息ID:2',quoted)
        context.current_turn=turn.model_copy(update={'metadata':{'conversation_type':'private'}})
        self.assertNotIn('QQ 多人群聊',agent._build_system_context(context))


class IntentDecisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session,self.turn,self.message=make_case({'current':'谁推荐个电影？'})
        self.agent=SimpleNamespace(client=object(),model='test-main')
        self.detector=SimpleNamespace(detect=AsyncMock(return_value=SimpleNamespace(should_reply=True,probability=.8,model='test-jev')),
            fallback=AsyncMock(return_value=True),reply_threshold=.5)

    async def decide(self,**extra):
        return await should_respond(message=self.message,session=self.session,current_turn=self.turn,
            detector=self.detector,agent=self.agent,timeout=.01,fallback_timeout=.01,**extra)

    async def test_jev_false_never_calls_fallback(self):
        self.detector.detect.return_value.should_reply=False
        self.assertFalse(await self.decide())
        self.detector.fallback.assert_not_awaited()

    async def test_jev_error_uses_main_model_fallback(self):
        self.detector.detect.side_effect=RuntimeError('模拟接口故障')
        self.assertTrue(await self.decide())
        self.detector.fallback.assert_awaited_once()
        self.assertIs(self.detector.fallback.await_args.kwargs['client'],self.agent.client)

    async def test_jev_timeout_uses_fallback(self):
        async def wait(**kwargs):
            await asyncio.Event().wait()
        self.detector.detect.side_effect=wait
        self.assertTrue(await self.decide())
        self.detector.fallback.assert_awaited_once()

    async def test_cancel_does_not_start_fallback(self):
        self.detector.detect.side_effect=asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.decide()
        self.detector.fallback.assert_not_awaited()

    async def test_failed_or_timed_out_fallback_stays_silent(self):
        self.detector.detect.side_effect=RuntimeError('JEV故障')
        self.detector.fallback.side_effect=ValueError('无效判断')
        self.assertFalse(await self.decide())
        async def wait(**kwargs):
            await asyncio.Event().wait()
        self.detector.fallback.side_effect=wait
        self.assertFalse(await self.decide())

    async def test_other_mention_is_judged_instead_of_forcing_reply(self):
        self.message.content.append(MentionContent(target_id=21))
        self.detector.detect.return_value.should_reply=False
        self.assertFalse(await self.decide())
        self.detector.detect.assert_awaited_once()
        self.detector.fallback.assert_not_awaited()

    async def test_fallback_strict_json_without_tools(self):
        create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_reply":false}'))]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        detector=IntentDetector(client=object(),reply_threshold=.5)
        self.assertFalse(await detector.fallback(client=client,model='test',session=self.session,current_turn=self.turn,message=self.message))
        self.assertNotIn('tools',create.await_args.kwargs)
        self.assertEqual(json.loads(create.await_args.kwargs['messages'][1]['content'])['current_message']['text'],self.message.text)
        for invalid in ['{"should_reply":"false"}','{"should_reply":1}','[]','not json']:
            create.return_value.choices[0].message.content=invalid
            with self.assertRaises(ValueError):
                await detector.fallback(client=client,model='test',session=self.session,current_turn=self.turn)

    async def test_detector_reads_actual_probability_at_threshold(self):
        client=SimpleNamespace(system_one=AsyncMock(return_value=SimpleNamespace(nouls={'should_reply':SimpleNamespace(noul=.5)},model='resolved-model')))
        detector=IntentDetector(client=client,reply_threshold=.5)
        result=await detector.detect(session=self.session,current_turn=self.turn,message=self.message)
        self.assertTrue(result.should_reply)
        self.assertEqual(result.model,'resolved-model')
        self.assertEqual(result.probability,.5)


class IntentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_quoted_human_does_not_bypass_judgment(self):
        manager=ContextManager(InMemoryContextStore())
        manager.record_user_message(message(1,user_id=21,group_id=30))
        incoming=message(2,group_id=30)
        incoming.content=[ReplyContent(message_id=1),TextContent(text='你报的哪家驾校？')]
        detector=SimpleNamespace(detect=AsyncMock(return_value=SimpleNamespace(should_reply=False)))
        agent=SimpleNamespace(run=AsyncMock())
        transport=SimpleNamespace(wait_connected=AsyncMock(),reply_text=AsyncMock(return_value=99))
        await handle_message(MessageJob(incoming,asyncio.get_running_loop().time()+1),context_manager=manager,
            agent=agent,intent_detector=detector,transport=transport,intent_timeout=.1)
        detector.detect.assert_awaited_once()
        agent.run.assert_not_awaited()
        transport.reply_text.assert_not_awaited()

    async def test_pipeline_uses_fallback_and_records_only_delivered_reply(self):
        manager=ContextManager(InMemoryContextStore())
        incoming=message(1,group_id=30)
        detector=SimpleNamespace(detect=AsyncMock(side_effect=RuntimeError()),fallback=AsyncMock(return_value=True))
        agent=SimpleNamespace(run=AsyncMock(return_value='发生什么啦？'),client=object(),model='main')
        transport=SimpleNamespace(wait_connected=AsyncMock(),reply_text=AsyncMock(return_value=99))
        await handle_message(MessageJob(incoming,asyncio.get_running_loop().time()+1),context_manager=manager,
            agent=agent,intent_detector=detector,transport=transport,intent_timeout=.1)
        detector.fallback.assert_awaited_once()
        agent.run.assert_awaited_once()
        history=next(iter(manager.store.sessions.values())).recent_messages
        self.assertEqual(history[-1].source_message_id,99)
        self.assertEqual(history[-1].role,ContextRole.ASSISTANT)

    async def test_restart_restores_identity_and_quoted_archived_bot(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            store=SQLiteContextStore(path/'history.sqlite3',path/'attachments')
            manager=ContextManager(store)
            first=message(1,group_id=30,mentioned=True)
            conversation,session,turn=manager.record_user_message(first)
            manager.record_assistant_message(conversation=conversation,session=session,content='先去公园。',timestamp=first.timestamp,source_message_id=501)
            session.summary.narrative='群里刚讨论了出游计划。'
            session.summary_through_seq=session.recent_messages[-1].seq
            session.recent_messages=[]
            manager.save_session(session)
            store.close()
            store=SQLiteContextStore(path/'history.sqlite3',path/'attachments')
            try:
                archived=store.history(first.conversation_id)
                self.assertEqual(archived[0]['metadata']['mentioned_ids'],[10])
                manager=ContextManager(store)
                followup=message(2,group_id=30)
                followup.content=[ReplyContent(message_id=501),TextContent(text='你刚才推荐的地方在哪？')]
                _,session,current=manager.record_user_message(followup)
                detector=SimpleNamespace(detect=AsyncMock(),fallback=AsyncMock())
                self.assertTrue(await should_respond(message=followup,session=session,current_turn=current,detector=detector,
                    agent=object(),timeout=.1,store=store))
                detector.detect.assert_not_awaited()
                self.assertIn('出游',session.summary.narrative)
                other=message(3,group_id=31)
                other.content=followup.content
                _,other_session,other_turn=manager.record_user_message(other)
                self.assertFalse(await should_respond(message=other,session=other_session,current_turn=other_turn,detector=None,
                    agent=object(),timeout=.1,store=store))
            finally:
                store.close()
