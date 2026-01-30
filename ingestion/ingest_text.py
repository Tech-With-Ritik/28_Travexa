from PyPDF2 import PdfReader
from docx import Document
from embeddings.text_embedder import embed_text

CHUNK_SIZE = 800
OVERLAP = 200

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def ingest_uploaded_text(file):
    text = ""

    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")

    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = " ".join(p.extract_text() or "" for p in reader.pages)

    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = " ".join(p.text for p in doc.paragraphs)

    chunks = chunk_text(text)

    embeddings, metadata = [], []

    for i, chunk in enumerate(chunks):
        embeddings.append(embed_text(chunk))
        metadata.append({
            "content": chunk,
            "source": file.name,
            "chunk": i,
            "modality": "text"
        })

    return embeddings, metadata
