"""Regression test: get_llm() must not pass deprecated sampling
parameters to the Gemini client.

``get_llm`` used to construct the client with ``temperature=0.2``
for model ``gemini-3.6-flash``, intending low-temperature, factual
answers. But Google deprecated ``temperature`` / ``top_p`` /
``top_k`` starting with gemini-3.6-flash: the API silently ignores
them, so the model actually answered at its default temperature
while the code claimed determinism — and Google says supplying
them will return an HTTP 400 in future model generations. The
parameters must not be passed at all; the prompt in
``generate_answer`` ("using ONLY the provided context") is what
keeps answers grounded.

Runs with the standard library only (``python3 -m unittest
tests.test_llm_sampling_params``); the heavy langchain/dotenv
dependencies are stubbed out, same as ``test_answer_list_content.py``.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RecordingLLM:
    """Stands in for ChatGoogleGenerativeAI, recording the kwargs
    it was constructed with so the test can assert on them."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        RecordingLLM.instances.append(self)


def _install_stubs():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: True

    lgg = types.ModuleType("langchain_google_genai")
    lgg.ChatGoogleGenerativeAI = RecordingLLM

    lcp = types.ModuleType("langchain_core.prompts")
    lcp.ChatPromptTemplate = MagicMock()
    lc_core = types.ModuleType("langchain_core")
    lc_core.prompts = lcp

    sys.modules.update({
        "dotenv": dotenv,
        "langchain_google_genai": lgg,
        "langchain_core": lc_core,
        "langchain_core.prompts": lcp,
    })


def _load_rag_chain():
    _install_stubs()
    for _mod in [m for m in list(sys.modules)
                 if m == "src.rag_chain" or m.startswith("src.rag_chain.")]:
        del sys.modules[_mod]
    import src.rag_chain as rag_chain
    return rag_chain


class TestLlmSamplingParams(unittest.TestCase):
    def setUp(self):
        RecordingLLM.instances.clear()

    def test_get_llm_passes_no_temperature(self):
        # Regression test: temperature=0.2 used to be passed to a
        # model that silently ignores it, so answers ran at the
        # default temperature despite the code claiming 0.2.
        rag_chain = _load_rag_chain()
        rag_chain.get_llm()

        self.assertEqual(len(RecordingLLM.instances), 1)
        self.assertNotIn(
            "temperature",
            RecordingLLM.instances[0].kwargs,
        )

    def test_get_llm_passes_no_deprecated_sampling_params(self):
        # top_p / top_k are deprecated and ignored for this model
        # family too; none of them may be passed.
        rag_chain = _load_rag_chain()
        rag_chain.get_llm()

        kwargs = RecordingLLM.instances[0].kwargs
        for param in ("temperature", "top_p", "top_k"):
            self.assertNotIn(param, kwargs)

    def test_get_llm_still_uses_the_flash_model(self):
        rag_chain = _load_rag_chain()
        rag_chain.get_llm()

        self.assertEqual(
            RecordingLLM.instances[0].kwargs.get("model"),
            "gemini-3.6-flash",
        )


if __name__ == "__main__":
    unittest.main()
