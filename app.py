from fastapi import FastAPI
from db.redis_client import r, create_index, index_exists
from utils.pdf_loader import load_pdf_text, chunk_text
from utils.embed import get_embeddings, rerank
from models.query_model import QueryRequest
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, REDIS_INDEX, TOP_K
import numpy as np
import time
from redis.commands.search.query import Query
from contextlib import asynccontextmanager
from utils.llm_client import ask_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing PDF + Redis setup...")

    # Startup code


    if not index_exists():
        full_text = load_pdf_text(PDF_PATH)
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = get_embeddings(chunks)
        vector_dim = len(embeddings[0])

        create_index(vector_dim)
        start = time.time()
        for i, emb in enumerate(embeddings):
            r.hset(f"chunk:{i}", mapping={
                "content": chunks[i],
                "embedding": np.array(emb, dtype=np.float16).tobytes()
            })
        print(f"✅ Inserted {len(chunks)} chunks into Redis in {time.time()-start:.2f}s")
    else:
        print("ℹ️ Redis index already exists — skipping embedding.")

    yield  # <- FastAPI now runs

    # Shutdown code
    print("Server is stopping...")

# Initialize app with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/query")
def query_pdf(request: QueryRequest):
    query = request.query

    q_emb = get_embeddings([query])[0]
    q_vector = np.array(q_emb, dtype=np.float16).tobytes()

    q = Query(f"*=>[KNN {TOP_K} @embedding $vector AS score]") \
        .return_fields("content", "score") \
        .sort_by("score", asc=True) \
        .paging(0, TOP_K)

    results = r.ft(REDIS_INDEX).search(q, query_params={"vector": q_vector})

    # Extract docs for reranking
    documents = [doc.content for doc in results.docs]

    context = "\n\n".join(documents)

    prompt = f"""
    Answer the question naturally in your own words **only** from the context below.
    If you don't find the answer, then say "The document does not contain this information."

    Context:
    {context}

    Question: {query}
    Answer:
    """

    answer = ask_llm(prompt)

    return {"answer": answer}

    # # Perform reranking
    # rerank_scores = rerank(request.query, documents)
    # reranked_results = sorted(
    #     zip(results.docs, rerank_scores),
    #     key=lambda x: x[1],
    #     reverse=True
    # )
    #
    # response = []
    # for rank, (doc, score) in enumerate(reranked_results, start=1):
    #     response.append({
    #         "rank": rank,
    #         "redis_score": float(doc.score),
    #         "rerank_score": float(score),
    #         "content": doc.content
    #     })
    #
    # return {"query": request.query, "results": response}
