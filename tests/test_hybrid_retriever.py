"""Regression test for HybridRetriever.invoke().

app.py calls ``vector_store.invoke(question)`` when the user asks a
question. HybridRetriever previously only defined ``search()``, so
every question crashed with:
    AttributeError: 'HybridRetriever' object has no attribute 'invoke'

Runs with the standard library only (``python3 -m unittest
tests.test_hybrid_retriever``); the heavy langchain/faiss/sentence-
transformers dependencies are stubbed out.

Also covers the empty-corpus guard: a PDF with no extractable text
(e.g. scanned / image-only) yields zero chunks, and building a
retriever from zero documents crashed with a cryptic IndexError
(from FAISS) / ZeroDivisionError (from BM25) instead of failing
clearly.
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


class TestCreateVectorStoreEmpty(unittest.TestCase):
    def test_empty_documents_raise_value_error(self):
        # Regression test: a text-less PDF produces zero chunks, and
        # create_vector_store([]) used to crash with a cryptic
        # IndexError (FAISS) / ZeroDivisionError (BM25). It must fail
        # with a clear ValueError so app.py can warn the user instead.
        with self.assertRaises(ValueError):
            create_vector_store([])


class TestHybridSearchIncludesKeywordHits(unittest.TestCase):
    """Regression test: the BM25 (keyword) leg must actually reach
    the answer.

    search() used to put every semantic hit first and then truncate
    to k, so whenever the corpus had >= k documents the keyword
    results were always cut off and BM25 never affected the answer
    at all — a document that was the top BM25 match for an exact
    keyword query was silently dropped.
    """

    def _retriever_with_fixed_bm25(self, docs, scores):
        """Build a retriever whose BM25 leg returns fixed scores.

        The retriever's ``bm25`` attribute is replaced on the
        instance (not the module global): sibling test modules evict
        and re-import ``src.*`` under their own stubs, so patching
        the module global can land on a different module object
        than the one ``create_vector_store`` was imported from.
        """

        class FixedBM25:
            def __init__(self, tokenized):
                pass

            def get_scores(self, query_tokens):
                return scores

        retriever = create_vector_store(docs)
        retriever.bm25 = FixedBM25([])
        return retriever

    def test_bm25_top_hit_is_returned(self):
        docs = [
            FakeDoc(f"Document number {i} about topic {i}.")
            for i in range(6)
        ]
        docs[5] = FakeDoc("Zebraxylophone is the secret keyword.")
        # Only doc 5 matches the query keyword.
        retriever = self._retriever_with_fixed_bm25(
            docs, [0, 0, 0, 0, 0, 10]
        )
        # Semantic search returns docs 0..4 only.
        retriever.vector_store.similarity_search.return_value = docs[:5]

        result = retriever.search("zebraxylophone", k=5)

        self.assertEqual(len(result), 5)
        self.assertIn(docs[5], result)

    def test_duplicate_hits_still_appear_once(self):
        # A document returned by both legs must not be duplicated.
        docs = [
            FakeDoc("The capital of France is Paris."),
            FakeDoc("Photosynthesis happens in chloroplasts."),
        ]
        retriever = self._retriever_with_fixed_bm25(docs, [5, 1])
        retriever.vector_store.similarity_search.return_value = docs

        result = retriever.search("capital of France", k=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(len({d.page_content for d in result}), 2)


if __name__ == "__main__":
    unittest.main()
