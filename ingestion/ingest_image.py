import tempfile
from embeddings.image_embedder import embed_image
from utils.ocr import extract_text_from_image

def ingest_uploaded_image(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        path = tmp.name

    emb = embed_image(path)
    text = extract_text_from_image(path) or "Image content"

    return [emb], [{
        "content": text,
        "source": file.name,
        "modality": "image"
    }]
