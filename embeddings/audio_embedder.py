from faster_whisper import WhisperModel
from embeddings.text_embedder import embed_text

model = WhisperModel("base", device="cpu", compute_type="int8")


def embed_audio(path):
    segments, _ = model.transcribe(path)
    text = " ".join(seg.text for seg in segments)

    embedding = embed_text(text)  # 🔥 reuses fixed text embedder
    return embedding, text
