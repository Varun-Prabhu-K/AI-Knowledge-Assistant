import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=api_key)


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

Format your answer using Markdown.
When the answer contains multiple items, use a numbered list.
When appropriate, use short paragraphs, bullet points, or numbered lists
to make the answer easy to read.

IMPORTANT SOURCE RULES:

1. Cite the supporting sources directly in your answer using the exact
   format [Source N], where N is one of the source numbers provided below.

2. Every factual part of your answer should have a supporting source citation.

3. If a statement is supported by multiple sources, cite all relevant
   sources, for example [Source 2, Source 5].

4. Do not invent source numbers.

5. Only cite sources that actually support the statement.

6. If the answer cannot be found in the provided context, say:
   "The information could not be found in the provided documents."
   In that case, return an empty source_ids list.

7. Do not include a separate "Sources" section in your answer.
   The application will generate that section automatically.

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
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "answer": {
                        "type": "STRING"
                    },
                    "source_ids": {
                        "type": "ARRAY",
                        "items": {
                            "type": "INTEGER"
                        }
                    }
                },
                "required": ["answer", "source_ids"]
            }
        )
    )

    result = json.loads(response.text)

    valid_source_ids = set(range(1, len(retrieved_chunks) + 1))

    source_ids = [
        source_id
        for source_id in result.get("source_ids", [])
        if source_id in valid_source_ids
    ]

    return result.get("answer", ""), source_ids