import os
from pathlib import Path

import streamlit as st

from src.ingestion import extract_text_from_pdf, create_chunks
from src.embeddings import create_embeddings
from src.vector_store import clear_collection, store_chunks
from src.retrieval import retrieve_chunks
from src.generation import generate_answer

def format_pages(pages):
    """
    Convert a list of page numbers into readable ranges.

    Example:
    [1, 2, 3, 5, 6] -> "1–3, 5–6"
    """

    if not pages:
        return "Unknown"

    pages = sorted(set(pages))

    ranges = []

    start = pages[0]
    end = pages[0]

    for page in pages[1:]:

        if page == end + 1:
            end = page

        else:

            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(
                    f"{start}–{end}"
                )

            start = page
            end = page

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(
            f"{start}–{end}"
        )

    return ", ".join(ranges)

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)


st.title("AI Knowledge Assistant")

st.write(
    "Upload one or more PDF documents and ask questions "
    "about their contents."
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF document(s)",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    if st.button("Process Documents"):

        with st.spinner(
            "Processing documents and building knowledge base..."
        ):

            # Remove previous knowledge
            clear_collection()

            total_chunks = 0

            for uploaded_file in uploaded_files:

                filename = Path(
                    uploaded_file.name
                ).name

                file_path = (
                    Path("documents") / filename
                )

                file_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with open(file_path, "wb") as file:
                    file.write(
                        uploaded_file.getbuffer()
                    )

                pages = extract_text_from_pdf(
                    str(file_path)
                )

                chunks = create_chunks(pages)

                if not chunks:
                    continue

                embeddings = create_embeddings(
                    chunks
                )

                store_chunks(
                    chunks,
                    embeddings
                )

                total_chunks += len(chunks)

            st.session_state["documents_processed"] = True

        st.success(
            f"Documents processed successfully. "
            f"Created {total_chunks} chunks."
        )


# --------------------------------------------------
# QUESTION FORM
# --------------------------------------------------

with st.form("question_form"):

    question = st.text_input(
        "Ask a question:"
    )

    submitted = st.form_submit_button(
        "Ask"
    )


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if submitted and question.strip():

    with st.spinner(
        "Searching the knowledge base..."
    ):

        retrieved_chunks = retrieve_chunks(
            question,
            top_k=10
        )

    if not retrieved_chunks:

        st.warning(
            "I could not find relevant information "
            "in the knowledge base."
        )

    else:

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks
            )

        st.subheader("Answer")

        st.write(answer)


        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        st.subheader("Sources")

        for number, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            source = chunk.get(
                "source",
                "Unknown source"
            )

            pages = chunk.get(
                "pages",
                []
            )

            formatted_pages = format_pages(
                pages
            ) if pages else "Unknown"

            st.write(
                f"**Source {number}:** "
                f"{source} — Pages {formatted_pages}"
            )