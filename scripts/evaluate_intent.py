"""用合成会话实测 JEV；不会连接 QQ。校准集选阈值，验证集只评估。"""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from qqbot.chat_context import direct_reply_decision, message_metadata
from qqbot.intent import INTENT_PROMPT_PATH, IntentDetector
from qqbot.schema import ContextMessage, ContextRole, ConversationType, MentionContent, MessageReceived, ReplyContent, Sender, Session, TextContent

ROOT = Path(__file__).resolve().parents[1]


def make_case(case):
    now = datetime(2026, 10, 17, 0, 0, tzinfo=timezone.utc)
    history = []
    rows = case.get('history', [])
    for i, (role, sender, text) in enumerate(rows, 1):
        history.append(ContextMessage(id=str(i), source_message_id=i, role=ContextRole(role),
            sender_id=sender if role == 'user' else None,
            sender_name=('同名成员' if case.get('same_names') else f'成员{sender}') if role == 'user' else None,
            content=text, timestamp=now-timedelta(seconds=case.get('gap_seconds', 0)+(len(rows)-i+1)*10),
            metadata={'bot_id':10, 'conversation_type':'group'}))
    content = [TextContent(text=case['current'])]
    content.extend(MentionContent(target_id=i) for i in case.get('mentions', []))
    if case.get('all'):
        content.append(MentionContent(is_all=True))
    if 'reply_to' in case:
        content.append(ReplyContent(message_id=case['reply_to']))
    message = MessageReceived(bot_id=10, conversation_id='qq:group:10:30', conversation_type=ConversationType.GROUP,
        group_id=30, message_id=1000, timestamp=now, sender=Sender(id=20,nickname='同名成员' if case.get('same_names') else '成员20'), content=content)
    current = ContextMessage(id='current', source_message_id=1000, role=ContextRole.USER, sender_id=20,
        sender_name=message.sender.display_name, timestamp=now, content=message.text, metadata=message_metadata(message))
    session = Session(session_id='simulation',conversation_id=message.conversation_id,created_at=now,updated_at=now,recent_messages=[*history,current])
    return session, current, message


def metrics(rows, threshold, *, with_rules=True):
    tp=fp=tn=fn=0
    for row in rows:
        predicted = row['rule'] if with_rules and row['rule'] is not None else row['probability'] >= threshold
        if row['expected']:
            tp += bool(predicted)
            fn += not predicted
        else:
            fp += bool(predicted)
            tn += not predicted
    return {'threshold':threshold,'tp':tp,'fn':fn,'fp':fp,'tn':tn,'weighted_errors':2*fn+fp}


async def main(args):
    load_dotenv(ROOT/'.env')
    cases = [c for c in json.loads((ROOT/'tests/fixtures/intent_cases.json').read_text()) if c['split']==args.split]
    if args.split != 'calibration' and args.threshold is None:
        raise SystemExit('验证集或补充检查必须传入已经由校准集确定的 --threshold')
    rows=[]
    async with AsyncTypeSafeClient(timeout=15) as client:
        detector=IntentDetector(client=client,reply_threshold=0.5)
        for case in cases:
            session,current,message=make_case(case)
            start=asyncio.get_running_loop().time()
            try:
                async with asyncio.timeout(20):
                    result=await detector.detect(session=session,current_turn=current,message=message)
            except Exception as exc:
                print(json.dumps({'id':case['id'],'error_type':type(exc).__name__}),flush=True)
                raise SystemExit('真实 JEV 请求失败，停止评估；未生成阈值') from None
            row={'id':case['id'],'expected':case['expected'],'probability':result.probability,
                'rule':direct_reply_decision(message,session),'model':result.model,
                'latency_ms':round((asyncio.get_running_loop().time()-start)*1000)}
            rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    candidates=[metrics(rows, i/100) for i in range(10,96,5)]
    threshold=args.threshold if args.threshold is not None else min(candidates,key=lambda x:(x['weighted_errors'],x['fp'],abs(x['threshold']-.5)))['threshold']
    report={'split':args.split,'synthetic':True,'prompt_sha256':hashlib.sha256(INTENT_PROMPT_PATH.read_bytes()).hexdigest(),
        'state_code_sha256':hashlib.sha256((ROOT/'src/qqbot/chat_context.py').read_bytes()).hexdigest(),
        'selection':'校准集最小化 2*漏回+误接；同分先少误接，再选最接近0.5的阈值',
        'selected':metrics(rows,threshold),'model_only':metrics(rows,threshold,with_rules=False),
        'threshold_comparison':candidates if args.split=='calibration' else [],'samples':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(args.output),'selected':report['selected']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['calibration','validation','audit'],required=True)
    parser.add_argument('--threshold',type=float)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
