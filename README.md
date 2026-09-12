# Research Paper Intelligence Assistant

## Overview
Answers questions about research papers by retrieving the most relevant passages from the source document and generating responses grounded in that retrieved evidence, rather than relying on the model's general knowledge alone. Aims to flag when a claim in the answer isn't clearly supported by the retrieved passages.

## Status
🚧 In development — document ingestion, chunking, embedding, and retrieval pipeline working.
LLM-based answer generation in progress.

## Tech Stack
- Python
- pypdf (PDF text extraction)
- sentence-transformers (local embeddings, all-MiniLM-L6-v2)
- ChromaDB (vector storage/retrieval)
- OpenRouter API (LLM generation) — coming in next update

## Current Capabilities
- Loads a PDF, splits it into chunks, embeds them, and retrieves the most relevant
  chunks for a given question using cosine similarity.
- Answer generation (using retrieved chunks + an LLM) not yet implemented.

## Setup
\`\`\`bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## Known Limitations
- Chunking is currently fixed-size (character-based), not sentence/paragraph-aware —
  can sometimes split relevant content awkwardly across chunk boundaries.
- PDF extraction can include some noise from figures/captions.
- Verifying whether an answer is truly grounded requires inspecting full retrieved
  passages, not truncated previews — truncated debug output can make correctly-grounded
  answers look unsupported.