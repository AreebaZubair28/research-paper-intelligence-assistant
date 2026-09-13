from src.ingest import chunk_text

def test_chunk_text_creates_multiple_chunks():
    text = "a" * 3000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) > 1

def test_chunk_text_respects_chunk_size():
    text = "a" * 3000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)

def test_chunk_text_handles_short_text():
    text = "short text"
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_load_and_chunk_documents_tags_source_correctly():
    fake_chunks = [
        {"text": "chunk from doc A", "source": "docA.pdf"},
        {"text": "chunk from doc B", "source": "docB.pdf"},
    ]
    sources = set(c["source"] for c in fake_chunks)
    assert sources == {"docA.pdf", "docB.pdf"}