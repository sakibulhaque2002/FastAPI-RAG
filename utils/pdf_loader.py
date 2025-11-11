import fitz

def load_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    full_text = " ".join([page.get_text().replace("\n", " ") for page in doc])
    return full_text

def chunk_text(text: str, max_words=1000, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks
