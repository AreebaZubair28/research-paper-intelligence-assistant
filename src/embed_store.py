from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

def build_collection(chunks, name="research_paper"):
    collection = client.create_collection(name=name)
    embeddings = model.encode(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def retrieve(collection, query, top_k=3):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results["documents"][0]