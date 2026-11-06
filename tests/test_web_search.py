import json
import unittest
from unittest.mock import patch

import httpx

from qqbot.tools.web_search import WebSearchTool


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_quota_has_explicit_non_retryable_result(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(432, json={'detail': {'error': 'usage limit'}})))
        with patch('qqbot.tools.web_search.httpx.AsyncClient', return_value=client):
            result = json.loads(await WebSearchTool(api_key='test').execute('{"query":"近期热点"}'))
        self.assertEqual(result['error'], 'search_quota_exceeded')
        self.assertIn('额度已用尽', result['message'])
        self.assertIn('不要重试', result['message'])

    async def test_news_request_preserves_sources_and_dates(self):
        def respond(request):
            body = json.loads(request.content)
            self.assertEqual((body['query'], body['topic'], body['days']), ('近期热点', 'news', 7))
            self.assertFalse(body['include_answer'])
            return httpx.Response(200, json={'results': [{'title': '测试新闻', 'url': 'https://example.invalid/news', 'content': '测试内容', 'published_date': '2026-10-28'}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch('qqbot.tools.web_search.httpx.AsyncClient', return_value=client):
            result = json.loads(await WebSearchTool(api_key='test').execute('{"query":"近期热点","topic":"news","days":7}'))
        self.assertEqual(result['result_count'], 1)
        self.assertEqual(result['results'][0]['url'], 'https://example.invalid/news')
        self.assertEqual(result['results'][0]['published_date'], '2026-10-28')
