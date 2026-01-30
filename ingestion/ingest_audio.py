import tempfile
from embeddings.audio_embedder import embed_audio

def ingest_uploaded_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        path = tmp.name

    emb, text = embed_audio(path)

    return [emb], [{
        "content": text,
        "source": file.name,
        "modality": "audio"
    }]
