"""Regression test: a corpus with no searchable words must not crash
the knowledge-base build.

A PDF whose extractable text holds no word characters at all — only
punctuation and symbols, e.g. a page of dashes, dots or bullet
separators ("--- ... !!!", "\u2022 \u2022 \u2022") — still yields chunks
from the text splitter, so the retriever's empty-documents guard
does not trigger. But every chunk tokenizes to an empty token list,
and the real ``rank_bm25.BM25Okapi`` dies building it:
``_calc_idf`` divides by ``len(self.idf)`` over an empty idf dict,
raising ``ZeroDivisionError`` out of ``HybridRetriever.__init__``.
The build then fails and the user is told the embedding model may
be unavailable — even though the embeddings are fine and the
semantic (FAISS) leg would work on the document.

A corpus with no tokens has no keyword leg to run: every BM25
score would be 0, so the keyword leg could contribute nothing
anyway. The retriever must skip BM25 in that case and answer from
the semantic leg alone, instead of crashing.

Runs with the standard library only (``python3 -m unittest
tests.test_bm25_no_tokens``); the heavy langchain/faiss/sentence-
transformers dependencies are stubbed out, and BM25 itself is
replaced by a fake that reproduces the real library's failure mode
(see ``RealisticBM25``).
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealisticBM25:
    """Stands in for rank_bm25.BM25Okapi, faithful to its failure
    mode: when the corpus contains no tokens at all, the real
    ``__init__`` dies in ``_calc_idf`` with ``ZeroDivisionError``
    (``idf_sum / len(self.idf)`` over an empty idf dict).

    ``get_scores`` raises if it is ever called: when the keyword
    leg was skipped, ``search()`` must never consult it.
    """

    instances = []

    def __init__(self, tokenized):
        if not any(tokenized):
            raise ZeroDivisionError("division by zero")
        self.tokenized_documents = tokenized
        RealisticBM25.instances.append(self)

    def get_scores(self, query_tokens):
        raise AssertionError(
            "keyword leg must not be consulted when it was skipped"
        )


def _install_stubs():
    lc_vs = types.ModuleType("langchain_community.vectorstores")
    lc_vs.FAISS = MagicMock()
    lc_community = types.ModuleType("langchain_community")
    lc_community.vectorstores = lc_vs
    lh = types.ModuleType("langchain_huggingface")
    lh.HuggingFaceEmbeddings = MagicMock()
    bm25_mod = types.ModuleType("rank_bm25")
    bm25_mod.BM25Okapi = RealisticBM25
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


class TestBm25NoTokens(unittest.TestCase):
    def setUp(self):
        RealisticBM25.instances.clear()

    def test_punctuation_only_corpus_builds_retriever(self):
        # Regression test: chunks with no word characters used to
        # crash create_vector_store with ZeroDivisionError from
        # BM25Okapi.__init__. The retriever must build.
        docs = [
            FakeDoc("--- ... !!! *** ... ---"),
            FakeDoc("\u2022 \u2022 \u2022 \u2022 \u2022"),
        ]
        retriever = create_vector_store(docs)
        self.assertIsNotNone(retriever)

    def test_search_uses_semantic_leg_only(self):
        # With no keyword leg, the answer must still come from the
        # semantic hits — and search() must never touch BM25.
        docs = [FakeDoc("--- ... !!! *** ... ---")]
        retriever = create_vector_store(docs)
        retriever.vector_store.similarity_search.return_value = docs

        result = retriever.search("---", k=5)

        self.assertEqual(result, docs)
        self.assertEqual(RealisticBM25.instances, [])

    def test_mixed_corpus_still_uses_bm25(self):
        # The skip must only trigger when NO document has a token:
        # one word-bearing document is enough for BM25 to work.
        docs = [
            FakeDoc("Paris is the capital of France."),
            FakeDoc("--- ... !!!"),
        ]
        retriever = create_vector_store(docs)
        self.assertEqual(len(RealisticBM25.instances), 1)


if __name__ == "__main__":
    unittest.main()
