import shutil
import tempfile
import unittest
from pathlib import Path

from docx import Document
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

from qqbot import documents
from qqbot.context import ContextManager
from qqbot.context.sqlite_store import SQLiteContextStore
from qqbot.schema import TextContent
from tests.helpers import message


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        self.addCleanup(lambda: self.store.close())
        self.manager = ContextManager(self.store)

    def test_restart_preserves_archives_and_summary_boundary(self):
        incoming = message(1)
        for i in range(1, 5):
            conv, session, turn = self.manager.record_user_message(message(i))
        session.summary.narrative = '旧消息的摘要'
        session.summary_through_seq = 2
        session.recent_messages = session.recent_messages[2:]
        self.manager.save_session(session)
        self.assertEqual(len(self.store.history(incoming.conversation_id)), 4)
        self.store.close()
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        restored = self.store.get_session(incoming.conversation_id)
        self.assertEqual([m.seq for m in restored.recent_messages], [3, 4])
        self.assertEqual(restored.summary.narrative, '旧消息的摘要')
        manager = ContextManager(self.store)
        _, session, last = manager.record_user_message(message(5))
        self.assertEqual(last.seq, 5)
        self.assertEqual(len(self.store.history(incoming.conversation_id)), 5)
        self.assertTrue(self.store.has_message(incoming.conversation_id, 1))
        tables = {r[0] for r in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'conversations', 'attachments'})

    def test_global_quota_cleans_old_conversation_not_idle_ttl(self):
        for user in (20, 21):
            for index in range(3):
                incoming = message(index, user_id=user)
                incoming.content = [TextContent(text='保留历史' * 60)]
                self.manager.record_user_message(incoming)
            self.store.release(message(user_id=user).conversation_id)
        before = self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0]
        self.store.conversation_limit = before - 1000
        self.store.cleanup()
        self.assertLessEqual(self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0], self.store.conversation_limit)
        self.assertEqual(len(self.store.history(message(user_id=21).conversation_id)), 3)
        self.manager.cleanup_expired()
        self.assertEqual(len(self.store.history(message(user_id=21).conversation_id)), 3)

    def test_one_active_conversation_can_evict_own_old_messages(self):
        self.store.conversation_limit = 2200
        for i in range(20):
            incoming = message(i)
            incoming.content = [TextContent(text='中' * 60)]
            with self.manager.processing(incoming):
                pass
        history = self.store.history(message().conversation_id)
        self.assertEqual(history[-1]['source_message_id'], 19)
        self.assertLess(len(history), 20)
        self.assertLessEqual(self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0], 2200)
        self.assertFalse(self.store.sessions)




class DocumentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    async def test_docx_and_excel_keep_locations_and_formulas(self):
        doc = Document()
        doc.add_paragraph('合同正文')
        doc.add_table(rows=1, cols=1).cell(0, 0).text = '条款表格'
        path = self.root / 'sample.docx'
        doc.save(path)
        parts = documents.extract(path, path.name, 100000)
        self.assertEqual([p['text'] for p in parts], ['合同正文', '条款表格'])
        book = Workbook()
        book.active['A1'] = '第一张表'
        book.active['B1'] = '=1+1'
        book.create_sheet('第二张')['C2'] = 123
        path = self.root / 'sample.xlsx'
        book.save(path)
        parts = documents.extract(path, path.name, 100000)
        self.assertIn('无缓存值', parts[0]['text'])
        self.assertEqual(parts[1]['sheet'], '第二张')
        self.assertIn('C2=123', parts[1]['text'])

    @unittest.skipUnless(shutil.which('soffice'), '未安装 LibreOffice')
    async def test_legacy_formats_and_office_visual_render(self):
        doc = Document()
        doc.add_paragraph('Document conversion fixture')
        original = self.root / 'fixture.docx'
        doc.save(original)
        work = self.root / 'legacy'
        work.mkdir()
        legacy = await documents.convert_office(original, original.name, 'doc', work, binary='soffice', timeout=30, limit=5000000)
        modern_work = self.root / 'modern'
        modern_work.mkdir()
        modern = await documents.convert_office(legacy, legacy.name, 'docx', modern_work, binary='soffice', timeout=30, limit=5000000)
        self.assertIn('fixture', documents.extract(modern, modern.name, 5000000)[0]['text'])
        pdf_work = self.root / 'pdf'
        pdf_work.mkdir()
        pdf = await documents.convert_office(legacy, legacy.name, 'pdf', pdf_work, binary='soffice', timeout=30, limit=5000000)
        self.assertFalse(documents.extract(pdf, pdf.name, 5000000)[0]['needs_ocr'])
        image, total = documents.render_pdf(pdf, 1, [0, 0, 0.5, 0.5])
        self.assertEqual(total, 1)
        self.assertTrue(image['image_url']['url'].startswith('data:image/jpeg'))

    @unittest.skipUnless(shutil.which('soffice'), '未安装 LibreOffice')
    async def test_xls_and_chart_render_include_hidden_sheet_outside_print_area(self):
        book = Workbook()
        sheet = book.active
        sheet.append(['Month', 'Value'])
        sheet.append(['Jan', 100])
        sheet.append(['Feb', 200])
        sheet['B4'] = '=SUM(B2:B3)'
        sheet['Z30'] = 'Outside print area'
        sheet.print_area = 'A1:B4'
        chart = BarChart()
        chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
        sheet.add_chart(chart, 'D2')
        hidden = book.create_sheet('Hidden')
        hidden['A1'] = 'Hidden sheet content'
        hidden.sheet_state = 'hidden'
        original = self.root / 'fixture.xlsx'
        book.save(original)
        async def convert(source, target, folder):
            work = self.root / folder
            work.mkdir()
            return await documents.convert_office(source, source.name, target, work, binary='soffice', timeout=30, limit=5000000)
        legacy = await convert(original, 'xls', 'legacy')
        modern = await convert(legacy, 'xlsx', 'modern')
        parts = documents.extract(modern, modern.name, 5000000)
        self.assertIn('缓存值=300', parts[0]['text'])
        self.assertEqual(parts[-1]['sheet'], 'Hidden')
        pdf = await convert(legacy, 'pdf', 'pdf')
        pages = documents.extract(pdf, pdf.name, 5000000)
        self.assertEqual(len(pages), 2)
        self.assertIn('Outside print area', pages[0]['text'])
        self.assertIn('Hidden sheet content', pages[1]['text'])
        image, total = documents.render_pdf(pdf, 1)
        self.assertEqual(total, 2)
        self.assertIn('image_url', image)
