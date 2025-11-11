import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from config import REDIS_HOST, REDIS_PORT, REDIS_INDEX

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

def index_exists():
    try:
        r.ft(REDIS_INDEX).info()
        return True
    except Exception:
        return False

def create_index(vector_dim: int):
    if not index_exists():
        r.ft(REDIS_INDEX).create_index(
            fields=[
                TextField("content"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {"TYPE": "FLOAT16", "DIM": vector_dim, "DISTANCE_METRIC": "COSINE"}
                )
            ],
            definition=IndexDefinition(prefix=["chunk:"], index_type=IndexType.HASH)
        )
        print("✅ Redis index created.")
