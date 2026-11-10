import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(uploaded_file):
    """Extract text from an uploaded PDF and split it into chunks."""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Debug information
        total_characters = sum(
            len(doc.page_content)
            for doc in documents
        )

        print("\n" + "=" * 50)
        print(f"PDF: {uploaded_file.name}")
        print(f"Pages extracted: {len(documents)}")
        print(f"Characters extracted: {total_characters}")
        print("=" * 50 + "\n")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Total chunks created: {len(chunks)}\n")

        return chunks

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)