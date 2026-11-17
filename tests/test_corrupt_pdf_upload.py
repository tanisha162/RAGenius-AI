"""Regression test: a corrupt/unreadable PDF must not crash the app.

Uploading a corrupt, empty or truncated PDF used to raise out of
``process_pdf`` (``PdfStreamError`` / ``EmptyFileError`` from pypdf)
with no ``try/except`` in ``app.py``, so the whole Streamlit app died
with a traceback instead of warning the user.

This test stubs Streamlit (and the heavy langchain/dotenv imports),
runs the real ``app.py`` top to bottom with a corrupt upload selected,
and asserts the app survives, warns naming the file, and keeps the
previous knowledge base instead of crashing.

Runs with the standard library only (``python3 -m unittest discover
-s tests``); the heavy dependencies are stubbed out, same as
``test_hybrid_retriever.py``.
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


class FakePdfError(Exception):
    """Stands in for pypdf's PdfStreamError / EmptyFileError."""


class RaisingLoader:
    """PyPDFLoader stub whose load() fails, as for a corrupt PDF."""

    def __init__(self, path):
        pass

    def load(self):
        raise FakePdfError("Stream has ended unexpectedly")


def _install_stubs():
    st = MagicMock()
    st.session_state = FakeSessionState()
    st.file_uploader.return_value = [
        FakeUpload("broken.pdf", b"this is not a pdf at all")
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
    lc_vs.FAISS = MagicMock()
    lc_dl = types.ModuleType("langchain_community.document_loaders")
    lc_dl.PyPDFLoader = RaisingLoader
    lc_community = types.ModuleType("langchain_community")
    lc_community.vectorstores = lc_vs
    lc_community.document_loaders = lc_dl
    lh = types.ModuleType("langchain_huggingface")
    lh.HuggingFaceEmbeddings = MagicMock()
    lts = types.ModuleType("langchain_text_splitters")
    lts.RecursiveCharacterTextSplitter = MagicMock()
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


# Install the stubs and run the real app.py top to bottom, with the
# corrupt upload selected and "Process Documents" pressed.
_ST = _install_stubs()
import app  # noqa: E402  (import runs the Streamlit script under stubs)


class TestCorruptPdfUpload(unittest.TestCase):
    def test_app_survives_corrupt_pdf(self):
        # The previous knowledge base (None here) is kept, not wiped
        # or left half-built by the failed upload.
        self.assertIsNone(_ST.session_state.get("vector_store"))

    def test_user_is_warned_naming_the_file(self):
        warnings = [str(call) for call in _ST.warning.call_args_list]
        self.assertTrue(
            any("broken.pdf" in w for w in warnings),
            f"expected a warning naming broken.pdf, got: {warnings}",
        )


if __name__ == "__main__":
    unittest.main()
