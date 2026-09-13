from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_and_chunk_documents(paths, chunk_size=1000, overlap=100):
    """Load multiple PDFs and return a list of (chunk_text, source_filename) tuples."""
    all_chunks = []
    for path in paths:
        text = load_pdf(path)
        chunks = chunk_text(text, chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": path})
    return all_chunks