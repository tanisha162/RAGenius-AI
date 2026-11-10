# 🧠 RAGenius AI — Context-Aware Document Q&A Platform

<div align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Storage-10B981?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/DeepSeek--R1-LLM_Inference-blue?style=for-the-badge" alt="DeepSeek" />
  <img src="https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge" alt="License" />

  <p align="center">
    <strong>Enterprise-Grade Retrieval-Augmented Generation (RAG) Architecture for Multi-Format Document Synthesis.</strong>
  </p>

</div>

---

## 📌 Executive Summary
**RAGenius AI** is an end-to-end Retrieval-Augmented Generation system designed to process, parse, index, and query unstructured documents at sub-second latencies. By combining intelligent recursive character chunking with high-dimensional vector embeddings, RAGenius AI delivers verifiable, context-grounded answers while completely mitigating hallucinations.

---

## ✨ Key Architectural Highlights
- **Multi-Format Ingestion Engine:** Automated parsing support for PDF, DOCX, TXT, and Markdown files.
- **Intelligent Chunking Pipeline:** Semantic text segmentation with configurable token overlaps to preserve semantic continuity across section boundaries.
- **Vector Embeddings & Indexing:** Low-latency vector persistence powered by **ChromaDB** with cosine similarity search.
- **State-of-the-Art LLM Integration:** Powered by **DeepSeek-R1** and OpenRouter APIs for logical reasoning and structured answers.
- **Asynchronous REST API:** Built with **FastAPI** featuring OpenAPI/Swagger specifications, streaming response capability, and clean endpoint separation.

---

## 🏗️ System Workflow Architecture

```text
[ Document Upload (PDF/DOCX/TXT) ]
              │
              ▼
[ Text Extraction & Normalization ]
              │
              ▼
[ Semantic Chunking & Overlap Windowing ]
              │
              ▼
[ Vector Ingestion (ChromaDB Vector Store) ]
              │
    ┌─────────┴─────────┐
    │ User Query Input  │
    └─────────┬─────────┘
              │
              ▼
[ Top-K Semantic Similarity Search ]
              │
              ▼
[ Augmented Context Prompt Injection ]
              │
              ▼
[ LLM Generation (DeepSeek-R1 / OpenRouter) ]
              │
              ▼
[ Contextual Answer + Source Citations ]
```

---

## 🛠️ Tech Stack & Dependencies
- **Core Language:** Python 3.10+
- **Backend API:** FastAPI, Uvicorn
- **Frontend / UI:** Streamlit
- **Embeddings & Vector Store:** ChromaDB, Sentence-Transformers
- **LLM Providers:** DeepSeek-R1, OpenRouter API
- **Utilities:** PyPDF, python-docx, NumPy

---

## 🚀 Quickstart & Setup Guide

### 1. Clone Repository
```bash
git clone https://github.com/tanisha162/RAGenius-AI.git
cd RAGenius-AI
```

### 2. Configure Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 5. Launch the Application
```bash
# To run the interactive UI:
streamlit run app.py
```

---

## 👩‍💻 Author & Connect
**Tanisha Gupta** — Full-Stack Developer & AI/ML Engineer  
- **GitHub:** [@tanisha162](https://github.com/tanisha162)  
- **LinkedIn:** [linkedin.com/in/tanisha-gupta162](https://www.linkedin.com/in/tanisha-gupta162)  
- **Email:** [gtanisha162@gmail.com](mailto:gtanisha162@gmail.com)
