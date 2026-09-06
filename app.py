import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_pdf
from src.vector_store import create_vector_store
from src.rag_chain import generate_answer

load_dotenv()


# Page configuration
st.set_page_config(
    page_title="RAGenius AI",
    page_icon="🤖",
    layout="wide"
)


# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# Header
st.title("🤖 RAGenius AI")
st.caption("AI-Powered Document Q&A Assistant")

st.write(
    "Upload a PDF, process it, and ask questions based on its content."
)


# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("🚀 Process Document", use_container_width=True):

            with st.spinner("Processing document..."):

                documents = process_pdf(uploaded_file)

                st.session_state.vector_store = create_vector_store(
                    documents
                )

            st.success("Document processed successfully!")


# Main area
st.divider()

question = st.text_input(
    "💬 Ask a question about your document",
    placeholder="Example: What is the main topic of this document?"
)


if question:

    if st.session_state.vector_store is None:

        st.warning("⚠️ Please upload and process a PDF first.")

    else:

        with st.spinner("🤖 RAGenius AI is thinking..."):

            documents = (
                st.session_state.vector_store.similarity_search(
                    question,
                    k=4
                )
            )

            answer = generate_answer(
                question,
                documents
            )

        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📚 Sources")

        for i, doc in enumerate(documents):

            page = doc.metadata.get("page", 0)

            with st.expander(
                f"Source {i + 1} — Page {page + 1}"
            ):
                st.write(doc.page_content)