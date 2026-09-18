import re
from pathlib import Path
import pymupdf


def clean_text(text):
    """
    Clean extracted PDF text by removing unnecessary
    whitespace while keeping the actual words.
    """

    # Replace multiple spaces, tabs and newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every non-empty page of a PDF.

    Each page is stored separately so that we can keep
    page numbers as metadata for later citations.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    source = Path(pdf_path).name

    for page_number, page in enumerate(document):
        text = page.get_text()

        text = clean_text(text)

        if text:
            pages.append({
                "text": text,
                "page": page_number + 1,
                "source": source
            })

    document.close()

    return pages


def create_chunks(pages, chunk_size=250, overlap=100):
    """
    Create chunks of approximately 250 words with 50 words
    of overlap.

    IMPORTANT:
    Chunks are NOT restricted to individual PDF pages.
    A chunk can contain text from multiple pages.

    Page numbers are preserved as metadata for citations.
    """

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    # This will contain:
    # (word, page_number, source)
    word_stream = []

    for page in pages:
        words = page["text"].split()

        for word in words:
            word_stream.append(
                (
                    word,
                    page["page"],
                    page["source"]
                )
            )

    chunks = []

    step = chunk_size - overlap

    chunk_number = 0

    for start in range(0, len(word_stream), step):

        current = word_stream[start:start + chunk_size]

        if not current:
            break

        words = [item[0] for item in current]

        pages_in_chunk = sorted(
            set(item[1] for item in current)
        )

        sources_in_chunk = sorted(
            set(item[2] for item in current)
        )

        chunk_text = " ".join(words)

        chunks.append({
            "text": chunk_text,
            "source": sources_in_chunk[0],
            "pages": pages_in_chunk,
            "chunk_id": chunk_number
        })

        chunk_number += 1

        # Stop once we have reached the end
        if start + chunk_size >= len(word_stream):
            break

    return chunks


if __name__ == "__main__":

    pdf_path = "documents/test.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(pages)} non-empty pages.")

    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks[:3]:

        print("\n--- Chunk ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Pages: {chunk['pages']}")
        print(chunk["text"][:500])