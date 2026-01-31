🧠 Multimodal RAG System

Secure, Explainable Multimodal Retrieval-Augmented Generation Platform

📌 Overview

This project is a Multimodal Retrieval-Augmented Generation (RAG) system built using Streamlit, FAISS, and Large Language Models (LLMs).
It enables users to ingest, retrieve, and reason over multiple data modalities through a unified conversational interface.

The system supports documents, images, audio, video, and Excel files, and generates evidence-grounded answers with confidence scoring, explainability, and voice output inside a secure authenticated environment.

✨ Key Features
-🔐 Authentication
-User Login & Signup
-Session-based access control
-User-isolated chat and data handling

💬 Conversation Management
-Multiple conversation sessions
-Create new conversations
-Rename conversations
-Delete conversations
-Export conversation history
-Automatic summarization of long conversations

⚠️ Platform Notes (Important)
*Live microphone input was intentionally removed
     Reason: PyAudio causes instability on Windows
     Voice input is supported via audio file upload
*Video ingestion uses MoviePy + FFmpeg
    Explicit resource cleanup added for Windows stability

🏆 Innovation & Uniqueness
-True multimodal RAG (not just text)
-Video → speech → semantic retrieval
-ChatGPT-style session memory inside RAG
-Confidence-aware answers
-Voice-enabled responses
-Excel-aware document ingestion
-Designed for real-world reliability, not just demos

💼 Potential Use Cases
-📚 Academic research assistant
-🏢 Enterprise document intelligence
-🎥 Video knowledge extraction
-📊 Business analytics over Excel + reports
-🧑‍⚖️ Legal / compliance document analysis
-🧠 Personal AI knowledge base

📈 Future Enhancements
-Persistent chat storage (database)
-Clickable video timestamps
-Role-based access control
-GPU-accelerated Whisper

🏗️ System Architecture

User
 │
 ▼
Streamlit User Interface
(Auth • Upload • Query • History)
 │
 ▼
Multimodal Ingestion Layer
(Text • Image • Audio • Video • Excel)
 │
 ▼
Chunking & Embedding
 │
 ▼
FAISS Vector Store
 │
 ▼
Retriever
 │
 ▼
LLM Generator
 │
 ▼
Answer + Confidence + Voice Output


📁 Project Structure

├── app.py                      # Main Streamlit application
├── auth/
│   └── auth_db.py              # User authentication logic
├── ingestion/
│   ├── ingest_text.py          # Text ingestion & chunking
│   ├── ingest_image.py         # Image ingestion
│   ├── ingest_audio.py         # Audio ingestion
│   └── ingest_excel.py         # Excel ingestion
├── embeddings/
│   └── text_embedder.py        # Embedding generation
├── vectorstore/
│   └── faiss_store.py          # FAISS vector database
├── retrieval/
│   ├── intent_classifier.py
│   └── confidence.py
├── rag/
│   └── generator.py            # LLM-based answer generation
├── utils/
│   └── export.py               # Report & history export
├── requirements.txt
└── README.md


🚀 How to Run the Project
1️⃣ Create Virtual Environment
python -m venv rag_env
rag_env\Scripts\activate   # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

-Cloud deployment (Docker / Streamlit Cloud)
-Cost & token usage analytics
