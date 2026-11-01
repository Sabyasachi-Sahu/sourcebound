"""Phase 5: hybrid retrieval (keyword + vector), metadata filters, pack routing,
federated search across packs, and reranking.

  python -m core.rag.retrieve "<question>" [--pack <pack>] [--no-rerank]
"""

import json
import math
import re
import sys
import time
from collections import Counter
from functools import lru_cache

from core.rag.common import (
    chat,
    embed,
    packs,
    read_jsonl,
    settings,
    tokens,
    utf8_stdout,
)
from core.rag.index import collection, store_file
from src.phase3.packs import load_pack


class BM25:
    """Okapi BM25 over the chunk store of one pack (rebuilt when the store changes)."""

    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.ids = [c["id"] for c in chunks]
        self.docs = [Counter(tokens(c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, allowed: set | None, limit: int) -> list:
        terms = set(tokens(query))
        scores = []
        for i, doc in enumerate(self.docs):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            score = 0.0
            for t in terms & doc.keys():
                f = doc[t]
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scores.append((score, self.ids[i]))
        scores.sort(reverse=True)
        return [cid for _, cid in scores[:limit]]


@lru_cache(maxsize=8)
def _store(pack: str, stamp: float):
    chunks = read_jsonl(store_file(pack))
    return {c["id"]: c for c in chunks}, BM25(chunks)


def store(pack: str):
    file = store_file(pack)
    if not file.exists():
        raise SystemExit(
            f"Pack {pack} is not ingested. Run: python -m core.rag.index ingest"
        )
    return _store(pack, file.stat().st_mtime)


def has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def route_packs(question: str) -> list:
    """Packs whose route keywords appear in the question (all packs if none)."""
    text = question.lower()
    hits = []
    for name, spec in packs().items():
        n = sum(has_phrase(text, k) for k in spec.get("route_keywords", []))
        if n:
            hits.append((n, name))
    hits.sort(key=lambda h: -h[0])
    chosen = [name for _, name in hits[: settings()["max_packs"]]]
    return chosen or list(packs())


def choose_scope(pack: str, question: str) -> str | None:
    """A scope filter inside a pack (records vs guidance, labels vs procedures)."""
    spec = packs()[pack]
    if "default_scope" not in spec:
        return None
    text = question.lower()
    for scope, words in spec.get("scope_keywords", {}).items():
        if any(has_phrase(text, w) for w in words):
            return scope
    return spec["default_scope"]


@lru_cache(maxsize=1)
def label_drugs() -> tuple:
    return tuple(load_pack("pharma").raw["source"]["drugs"])


def drug_filter(pack: str, question: str, scope: str | None) -> str | None:
    """For drug labels, keep only the named drug when exactly one is named."""
    if pack != "pharma" or scope != "labels":
        return None
    text = question.lower()
    named = [d for d in label_drugs() if has_phrase(text, d)]
    return named[0] if len(named) == 1 else None


def rrf(rankings: list, k: int) -> list:
    scores = Counter()
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] += 1 / (k + rank + 1)
    return [cid for cid, _ in scores.most_common()]


def in_force(chunk: dict, as_of: str | None) -> bool:
    """True if the chunk's document was effective on the as-of date (YYYY-MM-DD)."""
    effective = chunk.get("effective_date") or ""
    return as_of is None or not effective or effective <= as_of


def search_pack(
    pack: str,
    question: str,
    query_vec: list,
    use_filters: bool,
    as_of: str | None = None,
) -> dict:
    chunks, bm25 = store(pack)
    scope = choose_scope(pack, question) if use_filters else None
    drug = drug_filter(pack, question, scope) if use_filters else None
    allowed = {
        cid
        for cid, c in chunks.items()
        if (scope is None or c["scope"] == scope)
        and (drug is None or c["drug"] == drug)
        and in_force(c, as_of)
    }
    limit = settings()["candidates"]
    keyword = bm25.search(question, allowed, limit)
    conditions = [{"scope": scope}] if scope else []
    if drug:
        conditions.append({"drug": drug})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif conditions:
        where = {"$and": conditions}
    result = collection(pack).query(
        query_embeddings=[query_vec],
        n_results=min(limit, max(len(allowed), 1)),
        where=where,
        include=["distances"],
    )
    vector = [c for c in result["ids"][0] if c in allowed]
    similarity = {
        cid: 1 - d
        for cid, d in zip(result["ids"][0], result["distances"][0], strict=True)
        if cid in allowed
    }
    fused = rrf([keyword, vector], settings()["rrf_k"])
    return {
        "pack": pack,
        "scope": scope,
        "drug": drug,
        "fused": fused,
        "keyword": keyword,
        "vector": vector,
        "similarity": similarity,
    }


RERANK_PROMPT = """You rate how useful each excerpt is for answering the question.
Score 3 = directly answers it, 2 = partly answers it, 1 = same topic only,
0 = unrelated. Reply with JSON only: {"<excerpt id>": score, ...} for every id."""


def rerank(question: str, candidates: list, chunks: dict) -> dict:
    lines = [
        f"[{cid}] {' '.join(chunks[cid]['text'].split())[:500]}" for cid in candidates
    ]
    reply = chat(
        [
            {"role": "system", "content": RERANK_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n" + "\n".join(lines),
            },
        ]
    )
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        raw = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        raw = {}
    scores = {}
    for cid in candidates:
        try:
            scores[cid] = max(0.0, min(3.0, float(raw.get(cid, 0))))
        except (TypeError, ValueError):
            scores[cid] = 0.0
    return scores if raw else {}


def retrieve(
    question: str,
    pack_names: list | None = None,
    use_filters: bool = True,
    reranker: str | None = None,
    as_of: str | None = None,
) -> dict:
    """Return the top excerpts for a question, with timings and evidence strength."""
    timings = {}
    t0 = time.monotonic()
    routed = pack_names or route_packs(question)
    query_vec = embed([question], "query")[0]
    timings["embed_ms"] = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    per_pack = [search_pack(p, question, query_vec, use_filters, as_of) for p in routed]
    chunks = {}
    for p in routed:
        chunks.update(store(p)[0])
    merged = rrf([r["fused"] for r in per_pack], settings()["rrf_k"])
    similarity = {}
    for r in per_pack:
        similarity.update(r["similarity"])
    timings["search_ms"] = (time.monotonic() - t0) * 1000

    pool = merged[: settings()["rerank_pool"]]
    reranker = reranker or settings()["reranker"]
    scores = {}
    t0 = time.monotonic()
    if reranker == "llm" and pool:
        scores = rerank(question, pool, chunks)
    timings["rerank_ms"] = (time.monotonic() - t0) * 1000
    if scores:
        order = sorted(pool, key=lambda cid: (-scores[cid], pool.index(cid)))
        strength = max(scores.values()) / 3
        weak = max(scores.values()) < settings()["weak_rerank"]
    else:
        order = pool
        best = max((similarity.get(c, 0) for c in pool), default=0)
        strength, weak = best, best < settings()["weak_similarity"]

    top = select(order, chunks, routed, scores)
    return {
        "question": question,
        "packs": routed,
        "filters": {
            r["pack"]: {"scope": r["scope"], "drug": r["drug"]} for r in per_pack
        },
        "ids": top,
        "chunks": [chunks[c] for c in top],
        "scores": scores,
        "similarity": {c: round(similarity.get(c, 0), 3) for c in top},
        "strength": round(strength, 3),
        "weak": weak,
        "timings": timings,
        "candidates": len(merged),
    }


def select(order: list, chunks: dict, routed: list, scores: dict) -> list:
    """Top k, keeping a few excerpts from each routed pack for federated questions."""
    k = settings()["top_k"]
    if len(routed) < 2:
        return order[:k]
    quota = settings()["per_pack_min"]
    chosen = []
    for pack in routed:
        mine = [c for c in order if chunks[c]["pack"] == pack]
        if scores:
            mine = [c for c in mine if scores.get(c, 0) >= 1]
        chosen += mine[:quota]
    for cid in order:
        if len(chosen) >= k:
            break
        if cid not in chosen:
            chosen.append(cid)
    return sorted(chosen[:k], key=order.index)


def main():
    utf8_stdout()
    args = sys.argv[1:]
    pack = [args[args.index("--pack") + 1]] if "--pack" in args else None
    if pack:
        i = args.index("--pack")
        args = args[:i] + args[i + 2 :]
    reranker = "none" if "--no-rerank" in args else None
    args = [a for a in args if a != "--no-rerank"]
    if not args:
        raise SystemExit(__doc__)
    r = retrieve(" ".join(args), pack, reranker=reranker)
    print(f"Packs   : {r['packs']}   filters: {r['filters']}")
    print(
        f"Strength: {r['strength']}  weak: {r['weak']}  candidates: {r['candidates']}"
    )
    for c in r["chunks"]:
        score = r["scores"].get(c["id"], "-")
        print(
            f"  {c['id']:40} rerank {score}  sim {r['similarity'][c['id']]}  "
            f"{c['pack']}/{c['scope']}"
        )
    print({k: round(v) for k, v in r["timings"].items()})


if __name__ == "__main__":
    main()
