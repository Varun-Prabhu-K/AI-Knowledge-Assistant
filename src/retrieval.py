import ast

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)


collection = client.get_or_create_collection(
    name="knowledge_base"
)


# ---------------------------------------------------------
# Page metadata helper
# ---------------------------------------------------------

def parse_pages(pages):
    """
    Convert ChromaDB page metadata into a list of
    integer page numbers.
    """

    if pages is None:
        return []


    if isinstance(pages, int):
        return [pages]


    if isinstance(pages, float):
        return [int(pages)]


    if isinstance(pages, (list, tuple)):

        result = []

        for item in pages:
            result.extend(
                parse_pages(item)
            )

        return sorted(set(result))


    if isinstance(pages, str):

        pages = pages.strip()

        if not pages:
            return []


        # Handle strings such as:
        # "[12, 13]"
        # "(12, 13)"

        try:

            parsed = ast.literal_eval(pages)

            if parsed != pages:
                return parse_pages(parsed)

        except (ValueError, SyntaxError):
            pass


        # Handle:
        # "12,13,14"

        try:

            return sorted(
                set(
                    int(page.strip())
                    for page in pages.split(",")
                    if page.strip()
                )
            )

        except ValueError:
            return []


    return []


# ---------------------------------------------------------
# Convert Chroma result into our chunk format
# ---------------------------------------------------------

def make_chunk(document, metadata):
    """
    Convert a ChromaDB document and its metadata into
    the dictionary format used by the application.
    """

    if metadata is None:
        metadata = {}


    return {
        "text": document,

        "source": metadata.get(
            "source",
            "Unknown source"
        ),

        "pages": parse_pages(
            metadata.get("pages")
        )
    }


# ---------------------------------------------------------
# Retrieve chunks
# ---------------------------------------------------------

def retrieve_chunks(question, top_k=10):
    question_embedding = embedding_model.encode(question).tolist()

    # Get all stored documents and their metadata
    all_results = collection.get(
        include=["documents", "metadatas"]
    )

    all_documents = all_results.get("documents", [])
    all_metadatas = all_results.get("metadatas", [])

    # Find all unique source documents
    sources = sorted(
        set(
            metadata.get("source", "Unknown source")
            for metadata in all_metadatas
        )
    )

    # Map each stored document to its index
    document_to_index = {}

    for index, document in enumerate(all_documents):
        document_to_index[document] = index

    selected_indexes = set()

    # Retrieve relevant chunks separately from each document
    chunks_per_source = 3

    for source in sources:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(chunks_per_source, sum(
                1 for metadata in all_metadatas
                if metadata.get("source") == source
            )),
            where={"source": source},
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [[]])[0]

        for document in documents:
            if document not in document_to_index:
                continue

            index = document_to_index[document]

            selected_indexes.add(index)

            # Add neighboring chunks only from the same document
            if index > 0:
                previous_source = all_metadatas[index - 1].get("source")

                if previous_source == source:
                    selected_indexes.add(index - 1)

            if index + 1 < len(all_documents):
                next_source = all_metadatas[index + 1].get("source")

                if next_source == source:
                    selected_indexes.add(index + 1)

    retrieved_chunks = []

    for index in sorted(selected_indexes):
        metadata = all_metadatas[index]

        retrieved_chunks.append(
            make_chunk(
                all_documents[index],
                metadata
            )
        )

    return retrieved_chunks


# ---------------------------------------------------------
# Direct retrieval test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "How many research centres are "
        "mentioned in the document?"
    )


    chunks = retrieve_chunks(
        question,
        top_k=3
    )


    print(
        f"Retrieved {len(chunks)} chunks "
        "(including neighbours)."
    )


    for number, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\n--- Retrieved Chunk {number} ---"
        )


        print(
            f"Source: {chunk['source']}"
        )


        print(
            f"Pages: {chunk['pages']}"
        )


        print(
            chunk["text"][:500]
        )