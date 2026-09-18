import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()


# Read the Gemini API key.
api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# Create the Gemini client.
client = genai.Client(
    api_key=api_key
)


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using the user's question
    and the chunks retrieved from ChromaDB.
    """

    # Combine the retrieved chunks into one context.
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Pages: {chunk['pages']}\n"
            f"Content: {chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )


    # Tell Gemini to answer using the retrieved
    # document content.
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question primarily using
the provided document context.

If the answer cannot be found in the context,
say that the information could not be found
in the provided documents.

Do not invent facts.

User question:
{question}

Document context:
{context}
"""


    # Send the prompt to Gemini.
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    return response.text