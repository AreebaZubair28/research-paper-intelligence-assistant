from src.ingest import load_pdf, chunk_text
from src.embed_store import build_collection, retrieve
from src.generate import generate_answer

full_text = load_pdf("data/sample_paper.pdf")
chunks = chunk_text(full_text)
collection = build_collection(chunks)

question = "What is the purpose of Multi-Head Attention in the Transformer?"
retrieved_chunks = retrieve(collection, question, top_k=3)

result = generate_answer(question, retrieved_chunks)
print(f"Answer: {result['answer']}\n")
print("Sources used:")
for i, source in enumerate(result["sources"], 1):
    print(f"[{i}] {source}...")