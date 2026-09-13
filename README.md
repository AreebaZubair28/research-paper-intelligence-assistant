# Research Paper Intelligence Assistant

## Overview
Answers questions about research papers by retrieving the most relevant passages from the source document and generating responses grounded in that retrieved evidence, rather than relying on the model's general knowledge alone. Aims to flag when a claim in the answer isn't clearly supported by the retrieved passages.

## Status
🚧 In development — multi-document ingestion, chunking, embedding, retrieval, and
grounded answer generation all working. Refining unsupported-claim detection next.

## Tech Stack
- Python
- pypdf (PDF text extraction)
- sentence-transformers (local embeddings, all-MiniLM-L6-v2)
- ChromaDB (vector storage/retrieval)
- OpenRouter API (LLM generation) — coming in next update

## Current Capabilities
- Loads multiple PDFs, splits them into chunks, embeds them, and retrieves the most
  relevant chunks across all loaded documents for a given question using cosine similarity.
- Each retrieved chunk is tagged with its source document, so answers can be traced
  back to the correct paper.
- Generates evidence-grounded answers using retrieved chunks, with sources displayed
  alongside the answer.
- Tested on two unrelated papers (Transformer architecture, ResNet/residual learning)
  to confirm retrieval correctly distinguishes between documents rather than mixing
  content across them.

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