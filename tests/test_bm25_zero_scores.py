"""Regression test: the BM25 (keyword) leg must not inject documents
that matched no query keyword into the search results.

``HybridRetriever.search()`` used to take the top-k BM25 documents
regardless of score. When the query shared no token with any
document, BM25 scores every document 0, so the keyword leg returned
the first k documents in corpus order — arbitrary, irrelevant
documents that were interleaved into the results and displaced real
semantic hits (e.g. with 8 documents, a semantic leg returning docs
3..7 and an all-zero BM25 leg, the answer was built from
[3, 0, 4, 1, 5] instead of [3, 4, 5, 6, 7]).

Runs with the standard library only (``python3 -m unittest
tests.test_bm25_zero_scores``); the heavy langchain/faiss/sentence-
transformers dependencies are stubbed out, and BM25 itself is
replaced by a fake returning fixed scores (all-zero models the real
BM25Okapi behaviour for a query with no token overlap).
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

# Sibling test modules install their own stubs for ``src.*``; evict
# any cached copy so this module is imported against the stubs above
# (same pattern as test_bm25_tokenization.py).
for _mod in [m for m in list(sys.modules) if m.startswith("src")]:
    del sys.modules[_mod]

from src.vector_store import create_vector_store


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class FixedBM25:
    """Stands in for BM25Okapi, returning fixed scores."""

    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, query_tokens):
        return self.scores


class TestBm25ZeroScores(unittest.TestCase):
    def _retriever_with_scores(self, docs, scores):
        retriever = create_vector_store(docs)
        # Patch the instance, not the module global (same reason as
        # test_hybrid_retriever.py: sibling modules re-import src.*
        # under their own stubs).
        retriever.bm25 = FixedBM25(scores)
        return retriever

    def test_no_keyword_overlap_injects_nothing(self):
        # Regression test: with no keyword overlap anywhere (every
        # BM25 score 0), the keyword leg must contribute nothing —
        # the semantic hits must all survive.
        docs = [
            FakeDoc(f"Document number {i} about topic {i}.")
            for i in range(8)
        ]
        retriever = self._retriever_with_scores(docs, [0.0] * 8)
        retriever.vector_store.similarity_search.return_value = docs[3:8]

        result = retriever.search("flibberty gibbet unrelated", k=5)

        self.assertEqual(result, docs[3:8])

    def test_partial_overlap_keeps_only_matching_docs(self):
        # Only the document that actually matched a keyword may come
        # from the BM25 leg; zero-score documents must not.
        docs = [
            FakeDoc(f"Document number {i} about topic {i}.")
            for i in range(6)
        ]
        docs[5] = FakeDoc("Zebraxylophone is the secret keyword.")
        retriever = self._retriever_with_scores(
            docs, [0, 0, 0, 0, 0, 10]
        )
        retriever.vector_store.similarity_search.return_value = docs[:5]

        result = retriever.search("zebraxylophone", k=5)

        self.assertIn(docs[5], result)
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
