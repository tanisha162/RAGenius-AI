"""Regression test: a failed knowledge-base build must not crash
the app.

Clicking "Process Documents" calls ``create_vector_store`` to build
the hybrid retriever, and that step can fail — the embedding model
is downloaded on first use (so it needs network), and FAISS can
error on bad input. That used to propagate out of ``app.py`` with
no ``try/except``, so the whole Streamlit app died with a traceback
instead of warning the user, and any previous knowledge base was
left in an unknown state.

The app must survive, warn the user, and keep any previous
knowledge base (and its statistics) untouched.

Runs with the standard library only (``python3 -m unittest discover
-s tests``); the heavy dependencies are stubbed out, same as
``test_corrupt_pdf_upload.py``.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeSessionState(dict):
    """dict that also supports the attribute access Streamlit code uses
    (``st.session_state.vector_store = ...``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class FakeRetriever:
    """Stands in for a previously built knowledge base."""

    def invoke(self, query):
        return [FakeDoc("Paris is the capital of France.")]


class SelectiveLoader:
    def __init__(self, path):
        pass

    def load(self):
        return [FakeDoc("Some real extracted text about Paris.")]


class ExplodingFAISS:
    """Stands in for FAISS / the embedding model failing while the
    knowledge base is built (e.g. no network to download the
    sentence-transformers model on first run)."""

    @staticmethod
    def from_documents(documents, embeddings):
        raise RuntimeError(
            "sentence-transformers/all-MiniLM-L6-v2 unavailable"
        )


def _install_stubs():
    st = MagicMock()
    st.session_state = FakeSessionState()
    # A previous knowledge base is already in place.
    st.session_state.vector_store = FakeRetriever()
    st.session_state.document_count = 1
    st.session_state.chunk_count = 3
    st.file_uploader.return_value = [
        FakeUpload("good.pdf", b"%PDF-1.4 good content"),
    ]
    # Only the "Process Documents" button is pressed.
    st.button.side_effect = lambda *a, **k: bool(a) and (
        "Process Documents" in str(a[0])
    )
    st.columns.return_value = (MagicMock(), MagicMock())
    st.chat_input.return_value = ""

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: True

    lc_vs = types.ModuleType("langchain_community.vectorstores")
    lc_vs.FAISS = ExplodingFAISS
    lc_dl = types.ModuleType("langchain_community.document_loaders")
    lc_dl.PyPDFLoader = SelectiveLoader
    lc_community = types.ModuleType("langchain_community")
    lc_community.vectorstores = lc_vs
    lc_community.document_loaders = lc_dl
    lh = types.ModuleType("langchain_huggingface")
    lh.HuggingFaceEmbeddings = MagicMock()
    lts = types.ModuleType("langchain_text_splitters")
    splitter = MagicMock()
    splitter.split_documents.side_effect = lambda docs: docs
    lts.RecursiveCharacterTextSplitter = MagicMock(
        return_value=splitter
    )
    lgg = types.ModuleType("langchain_google_genai")
    lgg.ChatGoogleGenerativeAI = MagicMock()
    lcp = types.ModuleType("langchain_core.prompts")
    lcp.ChatPromptTemplate = MagicMock()
    lc_core = types.ModuleType("langchain_core")
    lc_core.prompts = lcp
    bm25_mod = types.ModuleType("rank_bm25")
    bm25_mod.BM25Okapi = MagicMock()

    sys.modules.update({
        "streamlit": st,
        "dotenv": dotenv,
        "langchain_community": lc_community,
        "langchain_community.vectorstores": lc_vs,
        "langchain_community.document_loaders": lc_dl,
        "langchain_huggingface": lh,
        "langchain_text_splitters": lts,
        "langchain_google_genai": lgg,
        "langchain_core": lc_core,
        "langchain_core.prompts": lcp,
        "rank_bm25": bm25_mod,
    })
    return st


# Install the stubs and run the real app.py top to bottom, with a
# good PDF uploaded, "Process Documents" pressed, and the vector
# store build raising. If the bug were still present, this import
# itself would raise the RuntimeError and the test module would
# fail to load.
_ST = _install_stubs()
for _mod in [m for m in list(sys.modules)
             if m == "app" or m.startswith("src")]:
    del sys.modules[_mod]
import app  # noqa: E402  (import runs the Streamlit script under stubs)


class TestVectorStoreFailure(unittest.TestCase):
    def test_app_survives_failed_knowledge_base_build(self):
        # Regression test: the app used to die with a traceback
        # when create_vector_store raised. Reaching this test at
        # all means the import above survived.
        self.assertIsNotNone(_ST.session_state.get("vector_store"))

    def test_user_is_warned_the_knowledge_base_failed(self):
        warnings = [str(call) for call in _ST.warning.call_args_list]
        self.assertTrue(
            any("knowledge base" in w.lower() for w in warnings),
            f"expected a warning about the knowledge base, got: {warnings}",
        )

    def test_no_success_message_for_failed_build(self):
        successes = [str(call) for call in _ST.success.call_args_list]
        self.assertFalse(
            any("successfully" in s for s in successes),
            f"expected no success message, got: {successes}",
        )

    def test_previous_knowledge_base_is_kept(self):
        # The failed build must not wipe the working retriever the
        # user already had.
        self.assertIsInstance(
            _ST.session_state.get("vector_store"), FakeRetriever
        )

    def test_previous_statistics_are_kept(self):
        # document_count / chunk_count must still describe the
        # previous knowledge base, not the failed batch.
        self.assertEqual(_ST.session_state.get("document_count"), 1)
        self.assertEqual(_ST.session_state.get("chunk_count"), 3)


if __name__ == "__main__":
    unittest.main()
