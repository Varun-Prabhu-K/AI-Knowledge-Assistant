import chromadb


def get_collection():
    """
    Get the persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="knowledge_base"
    )

    return collection


def clear_collection():
    """
    Remove all existing chunks from the knowledge base.
    """

    collection = get_collection()

    existing = collection.get()

    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)


def store_chunks(chunks, embeddings):
    """
    Store document chunks, embeddings and metadata in ChromaDB.
    """

    collection = get_collection()

    ids = [
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "pages": ",".join(map(str, chunk["pages"]))
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

    return collection


if __name__ == "__main__":

    from ingestion import extract_text_from_pdf, create_chunks
    from embeddings import create_embeddings

    pages = extract_text_from_pdf(
        "documents/test.pdf"
    )

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    collection = store_chunks(
        chunks,
        embeddings
    )

    print(
        f"Stored {collection.count()} chunks in ChromaDB."
    )