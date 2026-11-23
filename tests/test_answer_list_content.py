"""Regression test: a list-of-strings LLM response must not produce
a blank answer.

``generate_answer`` extracts text from the Gemini response. When the
model returns several content blocks, ``response.content`` is a list
— and langchain-core types ``AIMessage.content`` as
``str | list[str | dict]``, so that list may hold plain strings as
well as ``{"type": "text"}`` dicts. The old code only kept dict
items, so a list-of-strings response was silently dropped and the
user was shown an empty answer (with sources) instead of the answer
the model actually gave.

Runs with the standard library only (``python3 -m unittest
tests.test_answer_list_content``); the heavy langchain/dotenv
dependencies are stubbed out, same as ``test_llm_failure.py``.
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


class TestAnswerListContent(unittest.TestCase):
    def test_list_of_strings_is_kept(self):
        # Regression test: the model answered, but its content came
        # back as a list of plain strings — the answer must reach
        # the user, not come back blank.
        rag_chain = _load_rag_chain(FakeResponse([
            "Paris is the capital of France.",
            "It is in Europe.",
        ]))

        answer = rag_chain.generate_answer(
            "What is the capital of France?",
            [FakeDoc("Paris is the capital of France.")],
        )

        self.assertIn("Paris is the capital of France.", answer)
        self.assertIn("It is in Europe.", answer)

    def test_mixed_list_keeps_strings_and_text_blocks(self):
        rag_chain = _load_rag_chain(FakeResponse([
            {"type": "text", "text": "Paris is the capital."},
            "It is in Europe.",
        ]))

        answer = rag_chain.generate_answer(
            "What is the capital of France?",
            [FakeDoc("Paris is the capital of France.")],
        )

        self.assertIn("Paris is the capital.", answer)
        self.assertIn("It is in Europe.", answer)

    def test_dict_text_blocks_still_work(self):
        rag_chain = _load_rag_chain(FakeResponse([
            {"type": "text", "text": "Paris is the capital."},
        ]))

        answer = rag_chain.generate_answer(
            "What is the capital of France?",
            [FakeDoc("Paris is the capital of France.")],
        )

        self.assertEqual("Paris is the capital.", answer)


if __name__ == "__main__":
    unittest.main()
