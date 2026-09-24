# Research Paper Intelligence Assistant

A RAG-based assistant that answers questions from research papers using
retrieved evidence and identifies answers that are not sufficiently
supported by the source documents.

## Overview

The system retrieves relevant passages from uploaded research papers and
uses them as context for LLM-generated answers. A separate grounding check
then verifies whether the generated claims are supported by the retrieved
evidence.

## Status
🚧 In development — multi-document ingestion, chunking, embedding, retrieval, grounded
answer generation, and independent grounding verification all working and tested.

## Tech Stack
- Python
- pypdf (PDF text extraction)
- sentence-transformers (local embeddings, all-MiniLM-L6-v2)
- ChromaDB (vector storage/retrieval)
- OpenRouter API (LLM generation and grounding verification)

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
- Includes a second-pass grounding check that independently verifies whether the
  generated answer's claims are supported by the retrieved passages, rather than
  relying solely on the model's self-reported confidence.
- Tested on both answerable questions (correctly grounded, verified) and an
  intentionally unanswerable question (Vision Transformer learning rates, not
  covered by either source document) — correctly identified as unsupported in
  both the initial answer and the independent grounding check.

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

## Architecture

1. **Ingestion** (`src/ingest.py`) — loads PDF(s), extracts text, splits into overlapping
   character-based chunks, tags each chunk with its source document.
2. **Embedding + Storage** (`src/embed_store.py`) — converts each chunk into a vector
   using a local embedding model (`sentence-transformers`), stores vectors in a
   ChromaDB collection alongside source metadata.
3. **Retrieval** (`src/embed_store.py`) — embeds the user's question, finds the top-k
   most similar chunks across all loaded documents using cosine similarity.
4. **Generation** (`src/generate.py`) — sends retrieved chunks + the question to an LLM
   (via OpenRouter), instructed to answer using only the provided evidence, with a
   self-reported confidence rating.
5. **Grounding Check** (`src/generate.py`) — a second, independent LLM pass that verifies
   whether the generated answer's claims are actually supported by the retrieved
   passages, rather than relying solely on the model's self-assessment from step 4.

## Example

**Question:** "What problem does the degradation problem refer to, and how do residual
connections address it?"

**Answer:** The degradation problem refers to accuracy saturating and then degrading
rapidly as network depth increases — not caused by overfitting, but by increased training
error. Residual connections address this by reformulating stacked layers to learn a
residual mapping F(x) := H(x) - x instead of the original mapping directly, via shortcut
("skip") connections that add the input back to the output.

**Confidence:** Fully supported
**Sources:** 3 passages from "Deep Residual Learning for Image Recognition" (correctly
excluded the unrelated "Attention Is All You Need" paper also loaded in the same session)

See `EVALUATION.md` for full test cases, including a deliberately unanswerable question
used to verify the system correctly identifies when evidence is insufficient.
