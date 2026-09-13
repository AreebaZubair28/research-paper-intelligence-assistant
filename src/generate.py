import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "poolside/laguna-xs-2.1:free"  # swap if this 404s later — check openrouter.ai/collections/free-models

def generate_answer(question, retrieved_chunks_with_sources):
    context = "\n\n".join([
        f"[Passage {i+1}, source: {source}]: {chunk}"
        for i, (chunk, source) in enumerate(retrieved_chunks_with_sources)
    ])

    system_prompt = (
    "You are a research assistant. Answer the question using ONLY the provided passages. "
    "If the passages don't contain enough information to answer, say so explicitly. "
    "Do not use outside knowledge. "
    "At the end of your answer, add a line starting with 'Confidence:' rating how well "
    "the passages support your answer (Fully supported / Partially supported / Not supported)."
    )

    user_prompt = f"Passages:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved_chunks_with_sources   # now includes source filenames
    }

def check_grounding(answer, retrieved_chunks):
    context = "\n\n".join([f"[Passage {i+1}]: {chunk}" for i, (chunk, source) in enumerate(retrieved_chunks)])

    system_prompt = (
        "You are a fact-checking assistant. You will be given an answer and a set of "
        "source passages. Identify any specific claims in the answer that are NOT "
        "directly supported by the passages. For each unsupported claim, quote it and "
        "explain why it isn't backed by the passages. If everything is supported, say "
        "'All claims are supported by the provided passages.'"
    )
    user_prompt = f"Passages:\n{context}\n\nAnswer to check:\n{answer}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content