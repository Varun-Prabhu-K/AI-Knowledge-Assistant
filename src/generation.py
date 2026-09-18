import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=api_key)


def generate_answer(question, retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Pages: {chunk['pages']}\n"
            f"Content: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Do not invent facts or information that is not present in the documents.

If the answer is found in the documents, explain it clearly and concisely.

If the information cannot be found in the provided documents, say:
"The information could not be found in the provided documents."

User question:
{question}

Document context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text