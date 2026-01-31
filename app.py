import streamlit as st
import tempfile, os
import soundfile as sf
import pyttsx3
from langdetect import detect
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

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


# ===================== LOAD MODELS =====================
@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu")

whisper = load_whisper()


# ===================== VIDEO INGEST =====================
def ingest_uploaded_video(file):
    vectors, metas = [], []

    tmp_dir = tempfile.mkdtemp()   # ⬅ manual temp dir (important)
    video_path = os.path.join(tmp_dir, file.name)

    with open(video_path, "wb") as f:
        f.write(file.read())

    clip = None
    try:
        clip = VideoFileClip(video_path)

        audio_path = os.path.join(tmp_dir, "audio.wav")
        clip.audio.write_audiofile(audio_path, logger=None)

        segments, _ = whisper.transcribe(audio_path)

        speaker = 1
        last_end = 0

        for seg in segments:
            if seg.start - last_end > 1.5:
                speaker += 1

            text = seg.text.strip()
            if not text:
                continue

            vectors.append(embed_text(text))
            metas.append({
                "content": text,
                "source": file.name,
                "modality": "video",
                "speaker": f"Speaker {speaker}",
                "start": round(seg.start, 2),
                "end": round(seg.end, 2)
            })

            last_end = seg.end

    finally:
        # ✅ CRITICAL: release file handles
        if clip:
            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()

        # ✅ SAFE CLEANUP (Windows)
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except:
            pass

    return vectors, metas


# ===================== TEXT TO SPEECH =====================
def text_to_speech(text):
    try:
        lang = detect(text)
    except:
        lang = "en"

    engine = pyttsx3.init()

    for voice in engine.getProperty("voices"):
        try:
            if lang in voice.languages[0].decode().lower():
                engine.setProperty("voice", voice.id)
                break
        except:
            pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        path = f.name

    engine.save_to_file(text, path)
    engine.runAndWait()
    audio, sr_ = sf.read(path)
    return audio, sr_


# ===================== INIT =====================
create_users_table()
st.set_page_config(page_title="Multimodal RAG System", layout="wide")


# ===================== AUTH =====================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🔐 Login / Signup")

    t1, t2 = st.tabs(["Login", "Sign Up"])

    with t1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.authenticated = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with t2:
        u2 = st.text_input("New Username")
        p2 = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            if signup_user(u2, p2):
                st.success("Account created, please login")
            else:
                st.error("User already exists")

    st.stop()


# ===================== LOGOUT =====================
st.sidebar.write(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()


# ===================== CHAT STATE =====================
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1": {"messages": [], "summary": ""}
    }
    st.session_state.current_chat = "Chat 1"

if "store" not in st.session_state:
    st.session_state.store = FAISSStore()
    st.session_state.ingested = False


# ===================== SIDEBAR CHAT CONTROL =====================
st.sidebar.subheader("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    name = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[name] = {"messages": [], "summary": ""}
    st.session_state.current_chat = name

chat_names = list(st.session_state.chat_sessions.keys())
st.session_state.current_chat = st.sidebar.radio(
    "Select Chat",
    chat_names,
    index=chat_names.index(st.session_state.current_chat)
)

new_name = st.sidebar.text_input("✏️ Rename Chat")
if st.sidebar.button("Rename") and new_name:
    st.session_state.chat_sessions[new_name] = \
        st.session_state.chat_sessions.pop(st.session_state.current_chat)
    st.session_state.current_chat = new_name
    st.rerun()

if st.sidebar.button("🗑 Delete Chat"):
    if len(st.session_state.chat_sessions) > 1:
        del st.session_state.chat_sessions[st.session_state.current_chat]
        st.session_state.current_chat = list(st.session_state.chat_sessions)[0]
        st.rerun()

if st.sidebar.button("⬇️ Export Chat"):
    chat = st.session_state.chat_sessions[st.session_state.current_chat]
    content = f"Summary:\n{chat['summary']}\n\n"
    for q, a in chat["messages"]:
        content += f"Q: {q}\nA: {a}\n\n"

    st.sidebar.download_button(
        "Download",
        content,
        file_name=f"{st.session_state.current_chat}.txt"
    )


# ===================== MAIN UI =====================
st.title("🧠 Multimodal RAG System")
st.caption("ChatGPT-style Multimodal Assistant with History, Video & Voice Output")

files = st.file_uploader(
    "Upload files",
    type=[
        "pdf","txt","docx",
        "png","jpg","jpeg",
        "mp3","wav",
        "xls","xlsx",
        "mp4","mkv","avi"
    ],
    accept_multiple_files=True
)

if st.button("📥 Ingest Files"):
    st.session_state.store = FAISSStore()
    embeddings, metas = [], []

    with st.spinner("Indexing files..."):
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
            elif ext in ["mp4","mkv","avi"]:
                e,m = ingest_uploaded_video(f)
            else:
                continue

            embeddings.extend(e)
            metas.extend(m)

        st.session_state.store.add(embeddings, metas)
        st.session_state.ingested = True

    st.success("Files indexed successfully")


# ===================== CHAT DISPLAY =====================
chat = st.session_state.chat_sessions[st.session_state.current_chat]

if chat["summary"]:
    st.info(f"🧠 Chat Summary: {chat['summary']}")

for q, a in chat["messages"]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)


# ===================== QUERY INPUT =====================
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
            answer = "No relevant evidence found."
        else:
            evidence = [r[0] for r in results]
            conf = confidence_score(results, classify_intent(query))
            answer = generate_answer(query, evidence)
            answer += f"\n\nConfidence: {int(conf*100)}%"

    with st.chat_message("assistant"):
        st.write(answer)
        audio, sr_ = text_to_speech(answer)
        st.audio(audio, sample_rate=sr_)

    chat["messages"].append((query, answer))

    # ---------------- AUTO-SUMMARY ----------------
    if len(chat["messages"]) > 6:
        old = chat["messages"][:-3]
        text = "\n".join(f"Q:{q}\nA:{a}" for q, a in old)
        chat["summary"] = generate_answer(
            "Summarize this conversation briefly:",
            [{"content": text}]
        )
        chat["messages"] = chat["messages"][-3:]
