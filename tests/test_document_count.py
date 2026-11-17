"""Regression test: the "Documents" statistic must count only the
files that were actually indexed.

Uploading a mix of good and corrupt PDFs used to set
``st.session_state.document_count = len(uploaded_files)``, so the
sidebar claimed e.g. "2 Documents" when only 1 file had been read
into the knowledge base (the corrupt one was warned about and
skipped). ``document_count`` must be the number of files that were
successfully processed.

Runs with the standard library only (``python3 -m unittest discover
-s tests``); the heavy dependencies are stubbed out, same as
``test_corrupt_pdf_upload.py``. Because ``unittest discover`` imports
the test modules in one process, the cached ``app`` / ``src.*``
modules are evicted first so they rebind to this file's stubs.
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


class FakePdfError(Exception):
    """Stands in for pypdf's PdfStreamError / EmptyFileError."""


class SelectiveLoader:
    """PyPDFLoader stub that succeeds for a real PDF and fails for a
    corrupt one, mirroring what the real loader does."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            self._data = fh.read()

    def load(self):
        if self._data == b"this is not a pdf at all":
            raise FakePdfError("Stream has ended unexpectedly")
        return [FakeDoc("Some real extracted text about Paris.")]


def _install_stubs():
    st = MagicMock()
    st.session_state = FakeSessionState()
    st.file_uploader.return_value = [
        FakeUpload("good.pdf", b"%PDF-1.4 good content"),
        FakeUpload("broken.pdf", b"this is not a pdf at all"),
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


# Install the stubs and run the real app.py top to bottom, with one
# good and one corrupt upload selected and "Process Documents" pressed.
_ST = _install_stubs()
for _mod in [m for m in list(sys.modules)
             if m == "app" or m.startswith("src")]:
    del sys.modules[_mod]
import app  # noqa: E402  (import runs the Streamlit script under stubs)


class TestDocumentCount(unittest.TestCase):
    def test_counts_only_successfully_processed_files(self):
        # Regression test: document_count used to be
        # len(uploaded_files) == 2 even though broken.pdf was skipped.
        self.assertEqual(_ST.session_state.get("document_count"), 1)

    def test_chunk_count_matches_indexed_documents(self):
        self.assertEqual(_ST.session_state.get("chunk_count"), 1)

    def test_knowledge_base_built_from_good_file_only(self):
        self.assertIsNotNone(_ST.session_state.get("vector_store"))

    def test_corrupt_file_still_warned_about(self):
        warnings = [str(call) for call in _ST.warning.call_args_list]
        self.assertTrue(
            any("broken.pdf" in w for w in warnings),
            f"expected a warning naming broken.pdf, got: {warnings}",
        )


if __name__ == "__main__":
    unittest.main()
