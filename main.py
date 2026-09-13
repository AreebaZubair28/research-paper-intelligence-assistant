from src.ingest import load_and_chunk_documents
from src.embed_store import build_collection, retrieve
from src.generate import generate_answer

paths = ["data/Attention Is All You Need.pdf", "data/Deep Residual Learning for Image Recognition.pdf"]
chunk_dicts = load_and_chunk_documents(paths)
collection = build_collection(chunk_dicts)

question = "What problem does the degradation problem refer to, and how do residual connections address it?"
retrieved = retrieve(collection, question, top_k=3)

print(f"Total chunks created: {len(chunk_dicts)}")
print(f"Unique chunk texts: {len(set(c['text'] for c in chunk_dicts))}")

result = generate_answer(question, retrieved)
print(f"Answer: {result['answer']}\n")
print("Sources used:")
for i, (chunk, source) in enumerate(result["sources"], 1):
    print(f"[{i}] ({source}) {chunk[:150]}...")


from collections import Counter
source_counts = Counter(c["source"] for c in chunk_dicts)
print(source_counts)