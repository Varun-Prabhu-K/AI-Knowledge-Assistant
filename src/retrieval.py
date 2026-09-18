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

def retrieve_chunks(question, top_k=3):
    """
    Retrieve relevant chunks using vector similarity.

    After finding the most relevant chunks, also include
    their neighbouring chunks. This helps preserve context
    when an important section is split across chunk
    boundaries.
    """

    # -----------------------------------------------------
    # Convert question into an embedding
    # -----------------------------------------------------

    question_embedding = embedding_model.encode(
        question
    ).tolist()


    # -----------------------------------------------------
    # Find the most relevant chunks
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas"
        ]
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    # -----------------------------------------------------
    # Get every stored chunk
    # -----------------------------------------------------
    #
    # We need the complete collection so that we can find
    # the neighbours of the chunks selected above.

    all_results = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )


    all_documents = all_results.get(
        "documents",
        []
    )


    all_metadatas = all_results.get(
        "metadatas",
        []
    )


    # -----------------------------------------------------
    # Identify the indexes of the retrieved chunks
    # -----------------------------------------------------

    retrieved_indexes = []


    for document in documents:

        for index, stored_document in enumerate(
            all_documents
        ):

            if document == stored_document:

                retrieved_indexes.append(
                    index
                )

                break


    # -----------------------------------------------------
    # Add neighbouring chunks
    # -----------------------------------------------------

    selected_indexes = set()


    for index in retrieved_indexes:

        # Add the retrieved chunk itself.

        selected_indexes.add(index)


        # Add the chunk immediately before it.

        if index > 0:

            selected_indexes.add(
                index - 1
            )


        # Add the chunk immediately after it.

        if index + 1 < len(all_documents):

            selected_indexes.add(
                index + 1
            )


    # -----------------------------------------------------
    # Convert selected chunks
    # -----------------------------------------------------

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