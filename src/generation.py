import os
import json
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=api_key)


class AnswerResponse(BaseModel):
    answer: str = Field(
        description="The answer to the user's question, formatted using Markdown."
    )
    source_ids: List[int] = Field(
        description="The numbers of the sources that directly support the answer."
    )


def generate_answer(question, retrieved_chunks):
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {number}]\n"
            f"Document: {chunk['source']}\n"
            f"Pages: {chunk['pages']}\n"
            f"Content: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Do not invent facts.

Format the answer using Markdown.
When the answer contains multiple items, use a numbered list.
When appropriate, use short paragraphs, bullet points, or numbered lists
to make the answer easy to read.

SOURCE CITATION RULES:

1. Cite supporting sources directly in the answer using exactly:
   [Source N]

2. Every factual part of the answer should have a supporting source citation.

3. If a statement is supported by multiple sources, cite them like:
   [Source 2, Source 5]

4. Only use source numbers that actually exist in the provided context.

5. Do not invent source numbers, documents, or pages.

6. If the answer cannot be found in the provided context, say:
   "The information could not be found in the provided documents."

7. Do not create a separate Sources section. The application handles that.

User question:
{question}

Document context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AnswerResponse,
        ),
    )

    result = AnswerResponse.model_validate_json(response.text)

    valid_source_ids = set(range(1, len(retrieved_chunks) + 1))

    source_ids = [
        source_id
        for source_id in result.source_ids
        if source_id in valid_source_ids
    ]

    return result.answer, source_ids