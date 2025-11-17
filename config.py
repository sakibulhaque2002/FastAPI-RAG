import os
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_INDEX = "pdf_chunked_idx"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

PDF_PATH = "data/english.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3
