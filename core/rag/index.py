"""Phase 5: ingest each pack into its own ChromaDB collection, and re-index on change.

  python -m core.rag.index ingest [--pack <pack>] [--reset]
  python -m core.rag.index status

A manifest stores a hash per document. Only new or changed documents are
embedded again; removed documents are deleted from the collection. Changing
the chunk settings in rag.yaml changes every hash, so everything is rebuilt.

Step 5.12: every chunk also gets a hash of its own text, every document the hash
of its source file, and every ingest writes an immutable snapshot named by its
own hash (the index version). After ingest the answer cache is swept.
"""

import contextlib
import hashlib
import json
import sys
import time
from datetime import datetime

import chromadb
from chromadb.errors import NotFoundError

from core.rag.common import (
    embed,
    packs,
    path,
    read_jsonl,
    settings,
    utf8_stdout,
    write_jsonl,
)
from core.rag.documents import chunk_document, load_documents
from src.common.config import PROJECT_ROOT

CHUNK_KEYS = (
    "chunk_chars",
    "label_chunk_chars",
    "overlap_lines",
    "wrap_width",
    "embed_doc_prefix",
)


def store_file(pack: str):
    return path("store_dir") / "chunks" / f"{pack}.jsonl"


def manifest_file():
    return path("store_dir") / "manifest.json"


def read_manifest() -> dict:
    file = manifest_file()
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def collection_name(pack: str) -> str:
    return settings()["collection_prefix"] + pack


def client():
    return chromadb.PersistentClient(path=str(path("vector_dir")))


def collection(pack: str, create: bool = False):
    name = collection_name(pack)
    if create:
        return client().get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
    return client().get_collection(name)


def doc_hash(doc: dict) -> str:
    chunking = json.dumps({k: settings()[k] for k in CHUNK_KEYS}, sort_keys=True)
    body = "\n".join(doc["lines"]) + chunking + doc["id_style"] + doc["scope"]
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def file_hash(file) -> str:
    """Hash of the source file as it is on disk now (the freshness check)."""
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


def chroma_meta(chunk: dict) -> dict:
    keep = (
        "pack",
        "scope",
        "doc_id",
        "doc_type",
        "section",
        "section_no",
        "line_start",
        "line_end",
        "version",
        "drug",
        "synthetic",
    )
    return {k: chunk[k] for k in keep}


def ingest(pack: str, reset: bool = False) -> dict:
    started = time.monotonic()
    manifest = read_manifest()
    if reset:
        with contextlib.suppress(NotFoundError):
            client().delete_collection(collection_name(pack))
        manifest.pop(pack, None)
    old = manifest.get(pack, {})
    docs = load_documents(pack)
    coll = collection(pack, create=True)
    new_manifest, all_chunks, pending = {}, [], []
    added = changed = same = 0
    origin_hashes = {}
    for doc in docs:
        chunks = chunk_document(doc)
        for c in chunks:
            c["hash"] = text_hash(c["text"])
        all_chunks += chunks
        digest = doc_hash(doc)
        before = old.get(doc["doc_id"])
        origin = doc["origin"]
        if origin not in origin_hashes:
            origin_hashes[origin] = file_hash(origin)
        new_manifest[doc["doc_id"]] = {
            "hash": digest,
            "chunks": {c["id"]: c["hash"] for c in chunks},
            "source": doc["source"],
            "origin": origin.relative_to(PROJECT_ROOT).as_posix(),
            "file_hash": origin_hashes[origin],
            "version": str(doc.get("version", "")),
            "effective_date": str(doc.get("effective_date", "")),
            "reviewed": bool(doc.get("reviewed", True)),
        }
        if before and before["hash"] == digest:
            same += 1
            continue
        if before:
            changed += 1
            coll.delete(ids=list(before["chunks"]))
        else:
            added += 1
        pending += chunks
    for i in range(0, len(pending), 64):
        batch = pending[i : i + 64]
        texts = [c["text"] for c in batch]
        coll.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embed(texts, "doc"),
            metadatas=[chroma_meta(c) for c in batch],
        )
        print(
            f"  {pack}: embedded {min(i + 64, len(pending))} / {len(pending)}", end="\r"
        )
    if pending:
        print()
    removed = [d for d in old if d not in new_manifest]
    for doc_id in removed:
        coll.delete(ids=list(old[doc_id]["chunks"]))
    ids = [c["id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise SystemExit(f"Pack {pack}: duplicate chunk IDs, e.g. {dupes}")
    write_jsonl(store_file(pack), all_chunks)
    manifest[pack] = new_manifest
    manifest_file().write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    version = snapshot(manifest)
    return {
        "pack": pack,
        "documents": len(docs),
        "chunks": len(all_chunks),
        "added": added,
        "changed": changed,
        "unchanged": same,
        "removed": len(removed),
        "collection": coll.count(),
        "index_version": version,
        "seconds": round(time.monotonic() - started, 1),
    }


def current_file():
    return path("store_dir") / "current.json"


def current_version() -> str:
    file = current_file()
    if not file.exists():
        return "none"
    return json.loads(file.read_text(encoding="utf-8"))["index_version"]


def snapshot(manifest: dict) -> str:
    """Save the manifest as an immutable snapshot named by its own hash.

    Every answer records this index version, so any past answer can be traced
    to the exact document versions and chunk texts (by hash) behind it.
    """
    body = json.dumps(manifest, sort_keys=True)
    version = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    folder = path("snapshot_dir")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{version}.json"
    created = datetime.now().isoformat(timespec="seconds")
    if not target.exists():
        target.write_text(
            json.dumps(
                {"index_version": version, "created": created, "manifest": manifest}
            ),
            encoding="utf-8",
        )
    current_file().write_text(
        json.dumps({"index_version": version, "created": created}), encoding="utf-8"
    )
    return version


def status():
    manifest = read_manifest()
    print(
        f"{'pack':12} {'docs':>5} {'chunks':>7} {'indexed':>8} "
        f"{'unreviewed':>10}  check"
    )
    for pack in packs():
        entry = manifest.get(pack, {})
        chunks = len(read_jsonl(store_file(pack)))
        try:
            count = collection(pack).count()
        except NotFoundError:
            count = 0
        unreviewed = sum(not d["reviewed"] for d in entry.values())
        ok = (
            "OK"
            if entry and count == chunks
            else "NOT INGESTED"
            if not entry
            else "MISMATCH"
        )
        print(
            f"{pack:12} {len(entry):>5} {chunks:>7} {count:>8} {unreviewed:>10}  {ok}"
        )


def main():
    utf8_stdout()
    args = sys.argv[1:]
    if args[:1] == ["status"]:
        status()
        return
    if args[:1] != ["ingest"]:
        raise SystemExit(__doc__)
    chosen = [args[args.index("--pack") + 1]] if "--pack" in args else list(packs())
    from core.rag.freshness import sweep

    for pack in chosen:
        r = ingest(pack, reset="--reset" in args)
        flag = "OK" if r["collection"] == r["chunks"] else "MISMATCH"
        print(
            f"{pack:12} docs {r['documents']:>5}  chunks {r['chunks']:>6}  "
            f"new {r['added']:>4}  changed {r['changed']:>3}  "
            f"same {r['unchanged']:>5}  removed {r['removed']:>3}  "
            f"collection {r['collection']:>6} {flag}  ({r['seconds']}s)"
        )

    print(f"Index version: {current_version()}")
    counts = sweep()
    print(
        f"Cache sweep  : {counts['valid']} valid, {counts['rechecked']} re-checked "
        f"and kept, {counts['dropped']} dropped, {counts['expired']} expired"
    )


if __name__ == "__main__":
    main()
