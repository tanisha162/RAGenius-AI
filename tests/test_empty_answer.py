"""Regression test: a model response with no usable text must not
produce a blank answer — or the literal word "None".

``generate_answer`` extracts text from the Gemini response. When the
model returns no text at all — an empty content list, a list with no
text blocks (e.g. only thinking blocks from a reasoning model, or a
response blocked before any text was produced), an empty string, or
``content`` set to None — the old code returned "" (a blank answer,
shown to the user with sources and no explanation) or, for None,
the literal string "None" via ``str(response.content)``.

The answer must instead say plainly that the model returned an
empty response and to try again.

Runs with the standard library only (``python3 -m unittest
tests.test_empty_answer``); the heavy langchain/dotenv dependencies
are stubbed out, same as ``test_answer_list_content.py``.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeDoc:
    def __init__(self, content):
        self.page_content = content


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, *a, **k):
        pass


class FakeChain:
    """Stands in for ``prompt | llm``; returns the canned response."""

    def __init__(self, response):
        self._response = response

    def invoke(self, inputs):
        return self._response


class FakePrompt:
    def __init__(self, response):
        self._response = response

    def __or__(self, other):
        return FakeChain(self._response)


def _install_stubs(response):
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: True

    lgg = types.ModuleType("langchain_google_genai")
    lgg.ChatGoogleGenerativeAI = FakeLLM

    lcp = types.ModuleType("langchain_core.prompts")
    lcp.ChatPromptTemplate = MagicMock()
    lcp.ChatPromptTemplate.from_template.return_value = FakePrompt(
        response
    )
    lc_core = types.ModuleType("langchain_core")
    lc_core.prompts = lcp

    sys.modules.update({
        "dotenv": dotenv,
        "langchain_google_genai": lgg,
        "langchain_core": lc_core,
        "langchain_core.prompts": lcp,
    })


def _load_rag_chain(response):
    _install_stubs(response)
    for _mod in [m for m in list(sys.modules)
                 if m == "src.rag_chain" or m.startswith("src.rag_chain.")]:
        del sys.modules[_mod]
    import src.rag_chain as rag_chain
    return rag_chain


class TestEmptyAnswer(unittest.TestCase):
    def _answer(self, content):
        rag_chain = _load_rag_chain(FakeResponse(content))
        return rag_chain.generate_answer(
            "What is the capital of France?",
            [FakeDoc("Paris is the capital of France.")],
        )

    def test_empty_list_is_not_a_blank_answer(self):
        # Regression test: content=[] used to return "" — a blank
        # assistant message with sources and no explanation.
        answer = self._answer([])
        self.assertTrue(answer.strip())
        self.assertIn("empty response", answer)

    def test_none_content_is_not_the_word_none(self):
        # Regression test: content=None used to return the literal
        # string "None" as the answer via str(response.content).
        answer = self._answer(None)
        self.assertNotEqual(answer, "None")
        self.assertTrue(answer.strip())
        self.assertIn("empty response", answer)

    def test_thinking_only_blocks_are_not_a_blank_answer(self):
        # A reasoning model can return only thinking blocks and no
        # text block; that must not come back blank either.
        answer = self._answer([
            {"type": "thinking", "thinking": "Let me think..."},
        ])
        self.assertTrue(answer.strip())
        self.assertIn("empty response", answer)

    def test_empty_text_block_is_not_a_blank_answer(self):
        answer = self._answer([{"type": "text", "text": ""}])
        self.assertTrue(answer.strip())
        self.assertIn("empty response", answer)

    def test_empty_string_is_not_a_blank_answer(self):
        answer = self._answer("")
        self.assertTrue(answer.strip())
        self.assertIn("empty response", answer)

    def test_normal_answer_still_passes_through(self):
        answer = self._answer("Paris is the capital of France.")
        self.assertEqual("Paris is the capital of France.", answer)


if __name__ == "__main__":
    unittest.main()
