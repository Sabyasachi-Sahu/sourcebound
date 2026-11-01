"""Phase 3: embed a pack's documents and store them in its ChromaDB collection."""

import json
import sys

import chromadb
import ollama
from chromadb.errors import NotFoundError

from src.common.config import get_settings
from src.phase3.packs import DB_DIR, load_pack

BATCH_SIZE = 64


def load_docs(path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    args = sys.argv[1:]
    reset = "--reset" in args
    names = [a for a in args if not a.startswith("--")]
    if len(names) != 1:
        raise SystemExit(
            "Usage: python -m src.phase3.build_pack_index <pack> [--reset]"
        )
    pack = load_pack(names[0])
    if not pack.docs_file.exists():
        raise SystemExit(f"Documents file not found: {pack.docs_file}")

    s = get_settings()
    docs = load_docs(pack.docs_file)
    print(f"Pack             : {pack.name}")
    print(f"Loaded documents : {len(docs)}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(pack.collection)
            print(f"Deleted old collection '{pack.collection}'")
        except NotFoundError:
            print(f"No existing collection '{pack.collection}' to delete")

    collection = client.get_or_create_collection(
        name=pack.collection,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        response = ollama.embed(model=s.embed_model, input=[d["text"] for d in batch])
        payload = {
            "ids": [d["id"] for d in batch],
            "documents": [d["text"] for d in batch],
            "embeddings": response.embeddings,
        }
        metas = [d.get("metadata") for d in batch]
        if all(metas):
            payload["metadatas"] = metas
        collection.upsert(**payload)
        print(f"indexed {min(start + BATCH_SIZE, len(docs))} / {len(docs)}")

    print(f"Collection count : {collection.count()}")
    print(f"Stored at        : {DB_DIR}")


if __name__ == "__main__":
    main()
