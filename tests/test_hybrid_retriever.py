"""Regression test for HybridRetriever.invoke().

app.py calls ``vector_store.invoke(question)`` when the user asks a
question. HybridRetriever previously only defined ``search()``, so
every question crashed with:
    AttributeError: 'HybridRetriever' object has no attribute 'invoke'

Runs with the standard library only (``python3 -m unittest
tests.test_hybrid_retriever``); the heavy langchain/faiss/sentence-
transformers dependencies are stubbed out.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_stubs():
    lc_vs = types.ModuleType("langchain_community.vectorstores")
    lc_vs.FAISS = MagicMock()
    lc_community = types.ModuleType("langchain_community")
    lc_community.vectorstores = lc_vs
    lh = types.ModuleType("langchain_huggingface")
    lh.HuggingFaceEmbeddings = MagicMock()
    bm25_mod = types.ModuleType("rank_bm25")
    bm25_mod.BM25Okapi = MagicMock()
    sys.modules.update({
        "langchain_community": lc_community,
        "langchain_community.vectorstores": lc_vs,
        "langchain_huggingface": lh,
        "rank_bm25": bm25_mod,
    })


_install_stubs()

from src.vector_store import HybridRetriever, create_vector_store


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class TestHybridRetrieverInvoke(unittest.TestCase):
    def setUp(self):
        docs = [
            FakeDoc("The capital of France is Paris."),
            FakeDoc("Photosynthesis happens in chloroplasts."),
        ]
        self.retriever = create_vector_store(docs)
        # search() needs the stubbed FAISS store to return the docs.
        self.retriever.vector_store.similarity_search.return_value = docs

    def test_invoke_exists_and_returns_documents(self):
        result = self.retriever.invoke("What is the capital of France?")
        self.assertIsInstance(result, list)
        self.assertTrue(result)

    def test_invoke_matches_search(self):
        self.assertEqual(
            self.retriever.invoke("capital of France"),
            self.retriever.search("capital of France"),
        )


if __name__ == "__main__":
    unittest.main()
