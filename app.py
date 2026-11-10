import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_pdf
from src.vector_store import create_vector_store
from src.rag_chain import generate_answer

load_dotenv()


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="RAGenius AI",
    page_icon="🤖",
    layout="wide"
)


# ---------------- SESSION STATE ---------------- #

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.info(f"📄 {len(uploaded_files)} document(s) selected")

        if st.button(
            "🚀 Process Documents",
            use_container_width=True
        ):

            all_documents = []

            progress_bar = st.progress(0)

            with st.spinner("Processing documents..."):

                for index, uploaded_file in enumerate(uploaded_files):

                    documents = process_pdf(uploaded_file)

                    # Add source metadata
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    all_documents.extend(documents)

                    progress = int(
                        ((index + 1) / len(uploaded_files)) * 100
                    )

                    progress_bar.progress(progress)

                # Create Hybrid Retriever / Vector Store
                st.session_state.vector_store = create_vector_store(
                    all_documents
                )

                st.session_state.document_count = len(uploaded_files)
                st.session_state.chunk_count = len(all_documents)

            progress_bar.empty()

            st.success("Knowledge base created successfully! 🎉")

    st.divider()


    # ---------------- STATISTICS ---------------- #

    st.subheader("📊 Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documents",
            st.session_state.document_count
        )

    with col2:
        st.metric(
            "Chunks",
            st.session_state.chunk_count
        )

    st.divider()


    # ---------------- CLEAR CHAT ---------------- #

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# ---------------- HEADER ---------------- #

st.title("🤖 RAGenius AI")

st.caption(
    "AI-Powered Multi-Document Research Assistant"
)

st.write(
    "Upload one or multiple PDF documents and ask intelligent "
    "questions based on their content."
)


# ---------------- CHAT HISTORY ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ---------------- #

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Display User Message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Check if documents are processed
    if st.session_state.vector_store is None:

        with st.chat_message("assistant"):

            response = (
                "⚠️ Please upload and process your documents first."
            )

            st.markdown(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


    else:

        with st.chat_message("assistant"):

            with st.spinner("🤖 RAGenius AI is thinking..."):

                # IMPORTANT:
                # HybridRetriever uses invoke(), NOT similarity_search()
                documents = st.session_state.vector_store.invoke(
                    question
                )

                answer = generate_answer(
                    question,
                    documents
                )

            # Display Answer
            st.markdown(answer)


            # ---------------- SOURCES ---------------- #

            if documents:

                with st.expander("📚 View Sources"):

                    for i, doc in enumerate(documents):

                        source = doc.metadata.get(
                            "source",
                            "Unknown Document"
                        )

                        page = doc.metadata.get(
                            "page",
                            0
                        )

                        st.markdown(
                            f"### Source {i + 1}"
                        )

                        st.caption(
                            f"📄 {source} | Page {page + 1}"
                        )

                        st.write(
                            doc.page_content
                        )

                        st.divider()


        # Save Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )