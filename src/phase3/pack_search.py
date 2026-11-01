"""Phase 3: retrieve the closest items from any pack."""

import chromadb
import ollama

from src.common.config import get_settings
from src.phase3.packs import DB_DIR, Pack


def search(pack: Pack, question: str, top_k: int | None = None) -> list:
    s = get_settings()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(pack.collection)

    q_vec = ollama.embed(model=s.embed_model, input=question).embeddings[0]
    result = collection.query(query_embeddings=[q_vec], n_results=top_k or pack.top_k)

    hits = []
    for pid, text, dist in zip(
        result["ids"][0], result["documents"][0], result["distances"][0]
    ):
        hits.append({"id": pid, "text": text, "similarity": 1 - dist})
    return hits
