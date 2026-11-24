import unittest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
from langchain_openai import OpenAIEmbeddings
from src.observability.embeddings import track_embeddings
from src.eval.common import embedding_cost


class EmbeddingUsageTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        result={'data':[{'embedding':[1.0,2.0], 'index':0}],
                'usage':{'prompt_tokens':7,'total_tokens':7}}
        sync=Mock();sync.create.return_value=result
        async_client=SimpleNamespace(create=AsyncMock(return_value=result))
        embedder=OpenAIEmbeddings(api_key='offline', model='text-embedding-3-small',
                                  client=sync,async_client=async_client,
                                  chunk_size=1,check_embedding_ctx_length=False)
        traces=[]
        return track_embeddings(embedder,traces,'test'),traces

    async def test_real_langchain_sync_async_batch_usage(self):
        embedder,traces=self.make()
        self.assertEqual(embedder.embed_query('first'),[1.,2.])
        self.assertEqual(await embedder.aembed_documents(['second','third']),[[1.,2.],[1.,2.]])
        self.assertEqual(len(traces),3)
        self.assertEqual(sum(t['usage']['input_tokens'] for t in traces),21)
        self.assertTrue(all(t['usage_source']=='api_response' for t in traces))
        self.assertNotIn('second',str(traces))
        self.assertAlmostEqual(embedding_cost(traces,{'openai/text-embedding-3-small':{'input_per_million':1}}),21/1e6)

    async def test_errors_and_missing_usage_remain_unknown(self):
        embedder,traces=self.make()
        embedder.client.client.create.side_effect=ValueError('secret')
        with self.assertRaises(ValueError):embedder.embed_query('test')
        self.assertIsNone(traces[0]['usage'])
        self.assertNotIn('secret',str(traces))
        self.assertIsNone(embedding_cost(traces,{}))
        self.assertIsNone(embedding_cost(None,{}))
        self.assertEqual(embedding_cost([],{}),0)
