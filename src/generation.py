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

    response = client.interactions.create(
        model="gemini-3.8-flash",
        system_instruction=(
            "You answer questions using the provided document context. "
            "Stay grounded in that context and do not invent information."
        ),
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":
    test_question = "What counselling services are available to students?"

    from retrieval import retrieve_chunks

    results = retrieve_chunks(test_question)

    retrieved_chunks = []

    for i, document in enumerate(results["documents"][0]):
        retrieved_chunks.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "pages": results["metadatas"][0][i]["pages"]
        })

    answer = generate_answer(
        test_question,
        retrieved_chunks
    )

    print("\n--- Answer ---")
    print(answer)