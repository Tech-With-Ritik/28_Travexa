import pandas as pd
from embeddings.text_embedder import embed_text


def ingest_uploaded_excel(file, chunk_size=10):
    """
    Ingests Excel files (.xls, .xlsx)
    Converts rows into text chunks and embeds them
    """

    embeddings = []
    metadatas = []

    xls = pd.ExcelFile(file)

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # Drop completely empty rows
        df = df.dropna(how="all")

        if df.empty:
            continue

        # Convert rows to text
        rows_as_text = []
        for idx, row in df.iterrows():
            row_text = ", ".join(
                f"{col}: {str(row[col])}"
                for col in df.columns
                if pd.notna(row[col])
            )
            rows_as_text.append(
                f"Sheet: {sheet_name}, Row {idx} → {row_text}"
            )

        # Chunk rows
        for i in range(0, len(rows_as_text), chunk_size):
            chunk_rows = rows_as_text[i : i + chunk_size]
            chunk_text = "\n".join(chunk_rows)

            emb = embed_text(chunk_text)

            embeddings.append(emb)
            metadatas.append({
                "source": file.name,
                "sheet": sheet_name,
                "rows": f"{i}–{i+len(chunk_rows)-1}",
                "content": chunk_text,
                "modality": "excel"
            })

    return embeddings, metadatas
