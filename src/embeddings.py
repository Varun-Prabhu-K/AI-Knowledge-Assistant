from sentence_transformers import SentenceTransformer


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    from ingestion import extract_text_from_pdf, create_chunks

    pages = extract_text_from_pdf("documents/test.pdf")

    chunks = create_chunks(
        pages,
        source="test.pdf"
    )

    embeddings = create_embeddings(chunks)

    print(f"Created embeddings for {len(embeddings)} chunks.")
    print(f"Embedding dimensions: {len(embeddings[0])}")