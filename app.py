import streamlit as st

from vectorstore.faiss_store import FAISSStore
from embeddings.text_embedder import embed_text
from ingestion.ingest_text import ingest_uploaded_text
from ingestion.ingest_image import ingest_uploaded_image
from ingestion.ingest_audio import ingest_uploaded_audio

from retrieval.intent_classifier import classify_intent
from retrieval.confidence import confidence_score, uncertainty_message
from retrieval.conflict import detect_conflicts

from rag.generator import generate_answer

st.set_page_config(page_title="Multimodal RAG System")
st.title("🧠 Multimodal RAG System")

# ---------------- SESSION STATE ----------------
if "store" not in st.session_state:
    st.session_state.store = FAISSStore()
    st.session_state.ingested = False

store = st.session_state.store

# ---------------- FILE UPLOAD ----------------
files = st.file_uploader(
    "Upload files (PDF, DOCX, TXT, Images, Audio)",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "mp3", "wav"],
    accept_multiple_files=True
)

# ---------------- INGEST ----------------
if st.button("📥 Ingest Files"):
    st.session_state.store = FAISSStore()   # reset store
    store = st.session_state.store

    embeddings, metadatas = [], []

    for file in files:
        ext = file.name.split(".")[-1].lower()

        if ext in ["pdf", "txt", "docx"]:
            e, m = ingest_uploaded_text(file)
        elif ext in ["png", "jpg", "jpeg"]:
            e, m = ingest_uploaded_image(file)
        elif ext in ["mp3", "wav"]:
            e, m = ingest_uploaded_audio(file)
        else:
            continue

        embeddings.extend(e)
        metadatas.extend(m)

    if embeddings and len(embeddings) == len(metadatas):
        store.add(embeddings, metadatas)
        st.session_state.ingested = True
        st.success("✅ Files ingested successfully!")
    else:
        st.error("No valid content could be ingested.")

# ---------------- QUERY ----------------
query = st.text_input("Ask a question")

if st.button("Ask"):
    if not st.session_state.ingested:
        st.error("Please ingest files first.")
    else:
        intent = classify_intent(query)
        q_emb = embed_text(query)
        results = store.search(q_emb, k=5)

        if not results:
            st.warning("❌ No relevant evidence found.")
        else:
            evidence = [r[0] for r in results]

            # -------- CONFIDENCE --------
            confidence = confidence_score(results, intent)


            st.subheader("🔎 Confidence Level")
            st.progress(confidence)
            st.write(f"**Confidence:** {int(confidence * 100)}%")

            msg = uncertainty_message(confidence)
            if msg:
                st.warning(msg)

            # -------- CONFLICT DETECTION --------
            has_conflict, conflicts = detect_conflicts(evidence)

            if has_conflict:
                st.error("⚔️ Conflicting information detected!")
                for a, b in conflicts:
                    st.write("🔹 **Version A:**", a["content"][:300])
                    st.write("🔹 **Version B:**", b["content"][:300])

            # -------- GENERATION --------
            answer = generate_answer(query, evidence)
            st.success(answer)

            # -------- EVIDENCE --------
            st.subheader("📚 Evidence Used")
            for i, e in enumerate(evidence, start=1):
                st.write(f"{i}. **{e['source']}** ({e['modality']})")
