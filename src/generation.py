import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        [
            f"Source: {chunk['source']} | Pages: {chunk['pages']}\n"
            f"{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a knowledge assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the context does not contain enough information to answer
the question, say that the information was not found in the
provided documents.

Do not invent facts or use outside knowledge.

Context:

{context}

Question:

{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text