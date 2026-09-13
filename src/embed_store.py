from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

def build_collection(chunk_dicts, name="research_papers"):
    """chunk_dicts: list of {"text": ..., "source": ...} dicts"""
    collection = client.create_collection(name=name)
    texts = [c["text"] for c in chunk_dicts]
    sources = [c["source"] for c in chunk_dicts]
    embeddings = model.encode(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in sources],   # NEW: track source per chunk
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    return collection

def retrieve(collection, query, top_k=3):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))   # now returns (chunk, source) pairs