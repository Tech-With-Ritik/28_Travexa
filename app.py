import streamlit as st

from auth.auth_db import create_users_table, signup_user, login_user
from vectorstore.faiss_store import FAISSStore
from embeddings.text_embedder import embed_text

from ingestion.ingest_text import ingest_uploaded_text
from ingestion.ingest_image import ingest_uploaded_image
from ingestion.ingest_audio import ingest_uploaded_audio
from ingestion.ingest_excel import ingest_uploaded_excel

from retrieval.intent_classifier import classify_intent
from retrieval.confidence import confidence_score
from rag.generator import generate_answer
from utils.export import generate_report


# ---------------- INIT ----------------
create_users_table()
st.set_page_config(page_title="Multimodal RAG System", layout="wide")


# ---------------- AUTH ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.authenticated = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u2 = st.text_input("New Username")
        p2 = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            if signup_user(u2, p2):
                st.success("Account created")
            else:
                st.error("User exists")
    st.stop()

# ---------------- LOGOUT ----------------
st.sidebar.write(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()


# ---------------- CHAT SESSION INIT ----------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1": {"messages": [], "summary": ""}
    }
    st.session_state.current_chat = "Chat 1"

if "store" not in st.session_state:
    st.session_state.store = FAISSStore()
    st.session_state.ingested = False


# ---------------- SIDEBAR CHAT CONTROL ----------------
st.sidebar.subheader("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    name = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[name] = {"messages": [], "summary": ""}
    st.session_state.current_chat = name

chat_names = list(st.session_state.chat_sessions.keys())
st.session_state.current_chat = st.sidebar.radio(
    "Select Chat", chat_names,
    index=chat_names.index(st.session_state.current_chat)
)

# Rename chat
new_name = st.sidebar.text_input("✏️ Rename Chat")
if st.sidebar.button("Rename") and new_name:
    st.session_state.chat_sessions[new_name] = \
        st.session_state.chat_sessions.pop(st.session_state.current_chat)
    st.session_state.current_chat = new_name
    st.rerun()

# Delete chat
if st.sidebar.button("🗑 Delete Chat"):
    if len(st.session_state.chat_sessions) > 1:
        del st.session_state.chat_sessions[st.session_state.current_chat]
        st.session_state.current_chat = list(
            st.session_state.chat_sessions.keys()
        )[0]
        st.rerun()
    else:
        st.warning("At least one chat required")


# ---------------- MAIN UI ----------------
st.title("🧠 Multimodal RAG System")
st.caption("ChatGPT-style RAG with explainability")

# ---------------- FILE UPLOAD ----------------
files = st.file_uploader(
    "Upload documents",
    type=["pdf","txt","docx","png","jpg","jpeg","mp3","wav","xls","xlsx"],
    accept_multiple_files=True
)

if st.button("📥 Ingest Files"):
    st.session_state.store = FAISSStore()
    embeddings, metadatas = [], []

    with st.spinner("Indexing..."):
        for f in files:
            ext = f.name.split(".")[-1].lower()
            if ext in ["pdf","txt","docx"]:
                e,m = ingest_uploaded_text(f)
            elif ext in ["png","jpg","jpeg"]:
                e,m = ingest_uploaded_image(f)
            elif ext in ["mp3","wav"]:
                e,m = ingest_uploaded_audio(f)
            elif ext in ["xls","xlsx"]:
                e,m = ingest_uploaded_excel(f)
            else:
                continue
            embeddings.extend(e)
            metadatas.extend(m)

        st.session_state.store.add(embeddings, metadatas)
        st.session_state.ingested = True
    st.success("Files indexed")


# ---------------- CHAT DISPLAY ----------------
chat = st.session_state.chat_sessions[st.session_state.current_chat]

if chat["summary"]:
    st.info(f"🧠 Chat Summary: {chat['summary']}")

for q, a in chat["messages"]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)


# ---------------- QUERY INPUT ----------------
query = st.chat_input("Ask a question")

if query:
    with st.chat_message("user"):
        st.write(query)

    if not st.session_state.ingested:
        answer = "Please ingest files first."
    else:
        q_emb = embed_text(query)
        results = st.session_state.store.search(q_emb, k=6)

        if not results:
            answer = "No evidence found."
        else:
            evidence = [r[0] for r in results]
            conf = confidence_score(results, classify_intent(query))
            answer = generate_answer(query, evidence)
            answer += f"\n\nConfidence: {int(conf*100)}%"

    with st.chat_message("assistant"):
        st.write(answer)

    chat["messages"].append((query, answer))

    # ---------------- AUTO-SUMMARY ----------------
    if len(chat["messages"]) > 6:
        old = chat["messages"][:-3]
        text = "\n".join(f"Q:{q}\nA:{a}" for q,a in old)
        chat["summary"] = generate_answer(
            "Summarize this conversation briefly:", 
            [{"content": text}]
        )
        chat["messages"] = chat["messages"][-3:]


# ---------------- EXPORT CHAT ----------------
if st.sidebar.button("⬇️ Export Chat"):
    content = ""
    if chat["summary"]:
        content += f"Summary:\n{chat['summary']}\n\n"
    for q,a in chat["messages"]:
        content += f"Q: {q}\nA: {a}\n\n"

    st.sidebar.download_button(
        "Download",
        content,
        file_name=f"{st.session_state.current_chat}.txt",
        mime="text/plain"
    )
