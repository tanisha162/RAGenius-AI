import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
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
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts)

    return str(response.content)