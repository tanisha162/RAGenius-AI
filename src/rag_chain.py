import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_llm():
    # NOTE: do NOT pass temperature / top_p / top_k here. For
    # gemini-3.6-flash (and all later Gemini models) Google
    # deprecated those sampling parameters: the API silently
    # ignores them, so temperature=0.2 used to look like it made
    # answers more deterministic while the model actually ran at
    # its default temperature — and Google says supplying them
    # will return an HTTP 400 in future model generations. The
    # prompt in generate_answer ("using ONLY the provided
    # context") is what keeps answers grounded instead.
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def generate_answer(question, documents):

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = ChatPromptTemplate.from_template("""
You are RAGenius AI, an intelligent document assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, clearly say:
"I couldn't find this information in the uploaded document."

Give a clear, professional, well-formatted answer.

Context:
{context}

Question:
{question}

Answer:
""")

    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    # Extract clean text from Gemini response
    if isinstance(response.content, list):
        text_parts = []

        for item in response.content:
            if isinstance(item, str):
                # langchain-core types AIMessage.content as
                # ``str | list[str | dict]`` — a list may hold plain
                # strings as well as {"type": "text"} blocks. The old
                # code only kept dict items, so a list-of-strings
                # response was silently dropped and the user got a
                # blank answer.
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                # A text block's "text" can be None (or a non-string)
                # in a malformed / partial model response; appending
                # it as-is crashed "\n".join() with a TypeError, so
                # the whole answer step failed and the user was told
                # the model was unavailable even though it had
                # responded. Only real strings carry answer text.
                block_text = item.get("text", "")
                if isinstance(block_text, str):
                    text_parts.append(block_text)

        text = "\n".join(text_parts)
    elif response.content is None:
        # No content at all: str(None) would show the user the
        # literal word "None" as the answer.
        text = ""
    else:
        text = str(response.content)

    if not text.strip():
        # The model answered with no usable text — an empty content
        # list, a list with no text blocks (e.g. only thinking
        # blocks from a reasoning model, or a response blocked
        # before any text was produced), or an empty string. The
        # old code returned "" (a blank answer, shown with sources
        # and no explanation) or "None". Say so plainly instead.
        return (
            "I couldn't get an answer right now — the model returned "
            "an empty response. Please try asking again."
        )

    return text