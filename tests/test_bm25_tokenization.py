"""Regression test: BM25 keyword search must strip punctuation when
tokenizing documents and queries.

``HybridRetriever`` used to tokenize with ``str.split()``, which keeps
sentence punctuation attached to words: a document containing
"The secret keyword is zebraxylophone." produced the token
"zebraxylophone.", which never matched the query word
"zebraxylophone" — and a user question ending in "?" produced
"zebraxylophone?", which never matched the bare word in a document.
BM25 scored every document 0, so the keyword leg silently missed
exact keyword hits whenever the keyword ended a sentence or carried
punctuation, and the document was dropped from the results.

Runs with the standard library only (``python3 -m unittest
tests.test_bm25_tokenization``); the heavy langchain/faiss/sentence-
transformers dependencies are stubbed out, and BM25 itself is replaced
by a recording fake that captures the exact tokens it is handed, so
the test asserts on the tokenization rather than on BM25 internals.
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
    bm25_mod.BM25Okapi = RecordingBM25
    sys.modules.update({
        "langchain_community": lc_community,
        "langchain_community.vectorstores": lc_vs,
        "langchain_huggingface": lh,
        "rank_bm25": bm25_mod,
    })


class RecordingBM25:
    """Stands in for BM25Okapi, recording the tokens it receives.

    Scores rank the last document first so the keyword hit can be
    checked end to end through search().
    """

    instances = []

    def __init__(self, tokenized):
        self.tokenized_documents = tokenized
        self.query_tokens = None
        RecordingBM25.instances.append(self)

    def get_scores(self, query_tokens):
        self.query_tokens = query_tokens
        n = len(self.tokenized_documents)
        return [0.0] * (n - 1) + [10.0]


_install_stubs()

# Sibling test modules install their own stubs for ``src.*``; evict
# any cached copy so this module is imported against the stubs above
# (same pattern as test_llm_failure.py).
for _mod in [m for m in list(sys.modules) if m.startswith("src")]:
    del sys.modules[_mod]

from src.vector_store import create_vector_store


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class TestBm25Tokenization(unittest.TestCase):
    def setUp(self):
        RecordingBM25.instances.clear()
        self.docs = [
            FakeDoc(f"Document number {i} about topic {i}.")
            for i in range(5)
        ]
        # Keyword at the end of a sentence: its token used to carry
        # the trailing period ("zebraxylophone.").
        self.docs.append(
            FakeDoc("The secret keyword is zebraxylophone.")
        )
        self.retriever = create_vector_store(self.docs)
        # Semantic leg returns only docs 0..4 (keyword doc is not a
        # semantic hit), so the result depends on the BM25 leg.
        self.retriever.vector_store.similarity_search.return_value = (
            self.docs[:5]
        )

    def test_document_tokens_strip_sentence_punctuation(self):
        tokenized = RecordingBM25.instances[-1].tokenized_documents
        self.assertIn("zebraxylophone", tokenized[5])
        self.assertNotIn("zebraxylophone.", tokenized[5])

    def test_query_tokens_strip_question_mark(self):
        self.retriever.search("What is zebraxylophone?", k=5)
        query_tokens = RecordingBM25.instances[-1].query_tokens
        self.assertIn("zebraxylophone", query_tokens)
        self.assertNotIn("zebraxylophone?", query_tokens)

    def test_keyword_at_sentence_end_is_returned(self):
        result = self.retriever.search("zebraxylophone", k=5)
        self.assertIn(self.docs[5], result)


if __name__ == "__main__":
    unittest.main()
