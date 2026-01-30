import streamlit as st

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


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multimodal RAG System",
    layout="wide"
)

st.title("🧠 Multimodal RAG System")
st.caption(
    "Runtime Multimodal Retrieval-Augmented Generation with "
    "Confidence, Conflict Detection, Explainability, and Excel Support"
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
    type=[
        "pdf", "txt", "docx",
        "png", "jpg", "jpeg",
        "mp3", "wav",
        "xls", "xlsx"
    ],
    accept_multiple_files=True
)


# ---------------- INGEST FILES ----------------
if st.button("📥 Ingest Files"):
    if not files:
        st.warning("Please upload at least one file.")
    else:
        # Reset store for fresh ingestion
        st.session_state.store = FAISSStore()
        store = st.session_state.store
        st.session_state.chat_history = []

        embeddings = []
        metadatas = []

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

        st.success("✅ Files ingested and indexed successfully")


# ---------------- QUERY INPUT ----------------
st.divider()
st.subheader("❓ Ask a Question")

query = st.text_input(
    "Ask anything: summarize, compare documents, query Excel data, etc."
)


# ---------------- QUERY PROCESSING ----------------
if st.button("Ask"):
    if not st.session_state.ingested:
        st.error("Please ingest files before asking a question.")

    elif not query.strip():
        st.warning("Please enter a valid query.")

    else:
        intent = classify_intent(query)

        # -------- Scoped Conversation Memory --------
        memory_context = "\n".join(
            f"Q: {q}\nA: {a}"
            for q, a in st.session_state.chat_history[-3:]
        )

        full_query = (
            memory_context + "\n" + query
            if memory_context else query
        )

        q_embedding = embed_text(query)
        results = store.search(q_embedding, k=6)

        if not results:
            st.warning("⚠️ No relevant evidence found.")
        else:
            evidence = [r[0] for r in results]

            # ---------------- CONFIDENCE ----------------
            confidence = confidence_score(results, intent)
            st.subheader("📈 Confidence")
            st.progress(confidence)
            st.write(f"**Confidence:** {int(confidence * 100)}%")

            msg = uncertainty_message(confidence)
            if msg:
                st.warning(msg)

            # ---------------- DOCUMENT COVERAGE ----------------
            st.subheader("📊 Document Coverage")
            coverage = document_coverage(evidence)
            for doc, pct in coverage.items():
                st.write(f"**{doc}** → {pct}%")

            # ---------------- CONFLICT DETECTION ----------------
            has_conflict, conflicts = detect_conflicts(evidence)
            if has_conflict:
                st.error("⚔️ Conflicting evidence detected")

            # ---------------- GENERATE ANSWER ----------------
            st.subheader("🧠 Answer")
            answer = generate_answer(full_query, evidence)
            st.success(answer)

            # ---------------- SOURCE HIGHLIGHTING ----------------
            st.subheader("📚 Evidence Used (Exact Chunks)")
            for i, e in enumerate(evidence, 1):
                st.markdown(
                    f"""
**{i}. Source:** {e.get('source')}  
**Modality:** {e.get('modality')}  

> {e.get('content')[:500]}
"""
                )

            # ---------------- EXPORT REPORT ----------------
            report = generate_report(
                query=query,
                answer=answer,
                evidence=evidence,
                confidence=confidence
            )

            st.download_button(
                "⬇️ Export Report",
                data=report,
                file_name="multimodal_rag_report.txt",
                mime="text/plain"
            )

            # ---------------- SAVE MEMORY ----------------
            st.session_state.chat_history.append((query, answer))
