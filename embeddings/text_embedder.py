from sentence_transformers import SentenceTransformer
import numpy as np

# This model ALWAYS returns sentence embeddings (384-dim)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Returns a single sentence embedding of shape (384,)
    """

    if not text or not text.strip():
        return np.zeros(384, dtype="float32").tolist()

    emb = model.encode(
        text,
        normalize_embeddings=True,   # 🔥 important
        convert_to_numpy=True
    )

    # Safety check
    if emb.ndim != 1:
        raise ValueError(f"Text embedding is not 1D: shape={emb.shape}")

    return emb.astype("float32").tolist()
