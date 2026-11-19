"""Regression test: a failed answer step must not crash the app.

Asking a question calls the Gemini LLM, which can fail — a missing
or invalid GOOGLE_API_KEY, an exhausted quota, or a network error
all raise out of ``generate_answer``. That used to propagate out of
``app.py`` with no ``try/except``, so the whole Streamlit app died
with a traceback instead of answering, and the question was left
with no assistant reply at all.

This test stubs Streamlit (and the heavy langchain/dotenv imports),
runs the real ``app.py`` top to bottom with a question asked and the
LLM raising, and asserts the app survives, shows a friendly message
in the chat, and records that message in the chat history.

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


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {"source": "good.pdf", "page": 0}


class FakeRetriever:
    def invoke(self, query):
        return [FakeDoc("Paris is the capital of France.")]


def _install_stubs():
    st = MagicMock()
    st.session_state = FakeSessionState()
    st.session_state.vector_store = FakeRetriever()
    st.session_state.messages = []
    st.file_uploader.return_value = []
    # No button is pressed.
    st.button.side_effect = lambda *a, **k: False
    st.columns.return_value = (MagicMock(), MagicMock())
    st.chat_input.return_value = "What is the capital of France?"

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: True

    lc_vs = types.ModuleType("langchain_community.vectorstores")
    lc_vs.FAISS = MagicMock()
    lc_dl = types.ModuleType("langchain_community.document_loaders")
    lc_dl.PyPDFLoader = MagicMock()
    lc_community = types.ModuleType("langchain_community")
    lc_community.vectorstores = lc_vs
    lc_community.document_loaders = lc_dl
    lh = types.ModuleType("langchain_huggingface")
    lh.HuggingFaceEmbeddings = MagicMock()
    lts = types.ModuleType("langchain_text_splitters")
    lts.RecursiveCharacterTextSplitter = MagicMock()
    lgg = types.ModuleType("langchain_google_genai")

    class RaisingLLM:
        """Stands in for ChatGoogleGenerativeAI raising when the API
        key is missing/invalid, the quota is exhausted, or the
        network is down."""

        def __init__(self, *a, **k):
            raise ValueError(
                "Google API key not found. "
                "Please set GOOGLE_API_KEY."
            )

    lgg.ChatGoogleGenerativeAI = RaisingLLM
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
# question asked and the LLM raising. If the bug were still present,
# this import itself would raise the ValueError and the test module
# would fail to load.
_ST = _install_stubs()
for _mod in [m for m in list(sys.modules)
             if m == "app" or m.startswith("src")]:
    del sys.modules[_mod]
import app  # noqa: E402  (import runs the Streamlit script under stubs)


class TestLlmFailure(unittest.TestCase):
    def test_app_survives_llm_failure(self):
        # The question and the friendly fallback reply are both in
        # the chat history; the app did not die with a traceback.
        messages = _ST.session_state.get("messages", [])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_user_is_told_the_answer_is_unavailable(self):
        messages = _ST.session_state.get("messages", [])
        assistant_text = messages[1]["content"]
        self.assertIn("couldn't get an answer", assistant_text)
        self.assertIn("GOOGLE_API_KEY", assistant_text)

    def test_no_sources_shown_for_failed_answer(self):
        # The "View Sources" expander must not appear when there is
        # no answer to source.
        expander_calls = [
            str(call) for call in _ST.expander.call_args_list
        ]
        self.assertFalse(
            any("View Sources" in c for c in expander_calls),
            f"expected no sources expander, got: {expander_calls}",
        )


if __name__ == "__main__":
    unittest.main()
