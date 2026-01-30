import streamlit as st

from auth.auth_db import (
    create_users_table,
    signup_user,
    login_user
)

from vectorstore.faiss_store import FAISSStore
from embeddings.text_embedder import embed_text

from ingestion.ingest_text import ingest_uploaded_text
from ingestion.ingest_image import ingest_uploaded_image
from ingestion.ingest_audio import ingest_uploaded_audio
from ingestion.ingest_excel import ingest_uploaded_excel

from retrieval.intent_classifier import classify_intent
from retrieval.confidence import confidence_score, uncertainty_message
from retrieval.conflict import detect_conflicts
from retrieval.coverage import document_coverage

from rag.generator import generate_answer
from utils.export import generate_report


# ---------------- INIT AUTH DB ----------------
create_users_table()

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# ---------------- SESSION ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.authenticated:
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            if signup_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

    st.stop()


# ---------------- LOGOUT ----------------
st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


# ---------------- MAIN APP ----------------
st.title("🧠 Multimodal RAG System")
st.caption(
    "Secure, Explainable Multimodal Retrieval-Augmented Generation"
)

# ---------------- SESSION STATE ----------------
if "store" not in st.session_state:
    st.session_state.store = FAISSStore()
    st.session_state.ingested = False
    st.session_state.chat_history = []

store = st.session_state.store

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Files")

files = st.file_uploader(
    "Supported formats: PDF, DOCX, TXT, Images, Audio, Excel",
    type=["pdf","txt","docx","png","jpg","jpeg","mp3","wav","xls","xlsx"],
    accept_multiple_files=True
)

# ---------------- INGEST FILES ----------------
if st.button("📥 Ingest Files"):
    if not files:
        st.warning("Please upload at least one file.")
    else:
        st.session_state.store = FAISSStore()
        store = st.session_state.store
        st.session_state.chat_history = []

        embeddings, metadatas = [], []

        with st.spinner("Ingesting and indexing data..."):
            for file in files:
                ext = file.name.split(".")[-1].lower()

                if ext in ["pdf", "txt", "docx"]:
                    e, m = ingest_uploaded_text(file)
                elif ext in ["png", "jpg", "jpeg"]:
                    e, m = ingest_uploaded_image(file)
                elif ext in ["mp3", "wav"]:
                    e, m = ingest_uploaded_audio(file)
                elif ext in ["xls", "xlsx"]:
                    e, m = ingest_uploaded_excel(file)
                else:
                    continue

                embeddings.extend(e)
                metadatas.extend(m)

            store.add(embeddings, metadatas)
            st.session_state.ingested = True

        st.success("✅ Files ingested successfully")

# ---------------- QUERY ----------------
st.divider()
query = st.text_input("Ask a question")

if st.button("Ask"):
    if not st.session_state.ingested:
        st.error("Please ingest files first")
    else:
        intent = classify_intent(query)
        q_emb = embed_text(query)
        results = store.search(q_emb, k=6)

        if not results:
            st.warning("No evidence found")
        else:
            evidence = [r[0] for r in results]
            confidence = confidence_score(results, intent)

            st.progress(confidence)
            st.write(f"Confidence: {int(confidence*100)}%")

            answer = generate_answer(query, evidence)
            st.success(answer)

            report = generate_report(query, answer, evidence, confidence)
            st.download_button("⬇️ Export Report", report, "rag_report.txt")

            st.session_state.chat_history.append((query, answer))
