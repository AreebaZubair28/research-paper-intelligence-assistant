from src.ingest import load_and_chunk_documents
from src.embed_store import build_collection, retrieve
from src.generate import generate_answer, check_grounding

paths = ["data/Attention Is All You Need.pdf", "data/Deep Residual Learning for Image Recognition.pdf"]
chunk_dicts = load_and_chunk_documents(paths)
collection = build_collection(chunk_dicts)

question = "What optimizer learning rate schedule works best for training Vision Transformers?"
retrieved = retrieve(collection, question, top_k=3)

result = generate_answer(question, retrieved)
print(f"Answer: {result['answer']}\n")

grounding_check = check_grounding(result['answer'], result['sources'])
print(f"Grounding Check: {grounding_check}\n")

