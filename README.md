\# 🤖 RAGenius AI



An AI-powered Document Question Answering application built using \*\*Retrieval-Augmented Generation (RAG)\*\*.



RAGenius AI allows users to upload PDF documents and ask intelligent questions based on the document's content. The application retrieves relevant information using a vector database and generates context-aware answers using Google's Gemini AI.



\## 🚀 Features



\- 📄 Upload PDF documents

\- ✂️ Automatic text extraction and chunking

\- 🧠 Semantic search using HuggingFace embeddings

\- 🗄️ FAISS vector database for document retrieval

\- 🤖 AI-powered answers using Google Gemini

\- 🔍 Source-based responses

\- 💬 Interactive Streamlit interface



\## 🏗️ Architecture



```text

PDF Upload

&#x20;   ↓

Document Processing

&#x20;   ↓

Text Chunking

&#x20;   ↓

HuggingFace Embeddings

&#x20;   ↓

FAISS Vector Database

&#x20;   ↓

Similarity Search

&#x20;   ↓

Relevant Context

&#x20;   ↓

Gemini LLM

&#x20;   ↓

Answer + Sources

