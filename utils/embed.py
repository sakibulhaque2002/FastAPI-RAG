import numpy as np
from openai import OpenAI
from config import EMBEDDING_MODEL, RERANK_MODEL
from typing import Dict, Any

# Two vLLM servers
embedding_client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
reranker_client = OpenAI(base_url="http://localhost:8002", api_key="EMPTY")

def get_embeddings(texts):
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [np.array(r.embedding, dtype=np.float16) for r in response.data]

def rerank(query, docs):
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs
    }

    response = reranker_client.post(
        path="/v1/rerank",
        body=payload,
        cast_to=Dict[str, Any]
    )
    results = response["results"]
    scores = [item["relevance_score"] for item in results]
    return scores
