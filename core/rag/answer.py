"""Phase 5: answer a question from the documents, with exact citations,
supporting passages, verification, refusal, one self-correction, a cache and
per-stage latency.

Step 5.12: the cache is validated on every read (see core/rag/freshness.py),
high-risk packs cache evidence only, cited source files are re-hashed before an
answer is trusted, every answer records its index version, and --as-of answers
from the documents that were in force on a date.

  python -m core.rag.answer "<question>" [--pack <pack>] [--no-cache] [--judge]
                                         [--as-of YYYY-MM-DD]
"""

import hashlib
import json
import sys
import time
from datetime import datetime

from core.rag.common import (
    chat,
    chat_model,
    config_hash,
    packs,
    path,
    settings,
    utf8_stdout,
)
from core.rag.documents import citation_label
from core.rag.freshness import (
    get_evidence,
    live_chunks,
    lookup,
    refresh,
    risk_of,
    save_answer,
    save_evidence,
    stale_sources,
)
from core.rag.index import current_version
from core.rag.retrieve import retrieve, route_packs
from core.rag.verify import check, cited_ids
from src.phase3.packs import load_pack

REFUSAL = (
    "The documents provided do not contain enough information to answer this question."
)
BASE_PROMPT = """You answer questions using ONLY the excerpts in the user message.
Rules:
- After every sentence that states a fact, cite the excerpt ID in square brackets,
  copied exactly, for example [{example}]. Cite only IDs shown in the excerpts.
- Do not use outside knowledge. Do not guess numbers.
- If the excerpts do not answer the question, or answer only part of it, say
  clearly which part the documents do not contain.
- Excerpts may come from several packs (industries). Answer each part of the
  question from the relevant excerpts.
- The documents are synthetic or public test material. Do not give medical,
  legal or financial advice."""
REFORMULATE_PROMPT = """Rewrite the question as one short search query: keep the
key terms, add likely synonyms used in policies and procedures. Reply with the
query only."""


def cache_key(question: str, pack_list: list, judge: bool, as_of: str | None) -> str:
    """Identifies the question, not the data: validity is checked separately."""
    parts = [
        " ".join(question.lower().split()),
        ",".join(pack_list),
        str(judge),
        as_of or "",
        config_hash(),
        chat_model(),
        settings()["prompt_version"],
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def evidence_key(question: str, pack_list: list, as_of, reranker) -> str:
    parts = [
        " ".join(question.lower().split()),
        ",".join(pack_list),
        as_of or "",
        str(reranker),
        config_hash(),
        current_version(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def system_prompt(found: dict) -> str:
    example = found["ids"][0] if found["ids"] else "DOC-001.s01.01"
    prompt = BASE_PROMPT.format(example=example)
    if len(found["packs"]) == 1:
        pack = found["packs"][0]
        scope = found["filters"][pack]["scope"]
        if scope in packs()[pack].get("phase3_prompt_scopes", []):
            prompt += "\n\nPack instructions:\n" + load_pack(pack).raw["system_prompt"]
    return prompt


def context(found: dict) -> str:
    blocks = []
    for c in found["chunks"]:
        blocks.append(f"[{c['id']}] ({citation_label(c)})\n{c['text']}")
    return "\n\n".join(blocks)


def log_latency(entry: dict):
    file = path("store_dir") / "latency.jsonl"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def find_evidence(question, routed, as_of, reranker, risk, use_cache) -> tuple:
    """Retrieve (with one self-correction). High risk: reuse evidence of this index."""
    key = evidence_key(question, routed, as_of, reranker)
    if use_cache and risk == "high":
        cached = get_evidence(key)
        if cached:
            return cached, None, {"embed_ms": 0.0, "search_ms": 0.0}, True
    found = retrieve(question, routed, reranker=reranker, as_of=as_of)
    timings = dict(found["timings"])
    corrected = None
    if found["weak"] and settings()["retry_on_weak"]:
        t0 = time.monotonic()
        query = chat(
            [
                {"role": "system", "content": REFORMULATE_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]
        ).strip()
        second = retrieve(
            query, routed, use_filters=False, reranker=reranker, as_of=as_of
        )
        timings["retry_ms"] = (time.monotonic() - t0) * 1000
        corrected = {
            "query": query,
            "strength_before": found["strength"],
            "strength_after": second["strength"],
        }
        if second["strength"] > found["strength"]:
            found = second
    if use_cache and risk == "high":
        save_evidence(key, found)
    return found, corrected, timings, False


def ask(
    question: str,
    pack_list: list | None = None,
    use_cache: bool | None = None,
    judge: bool = False,
    reranker: str | None = None,
    as_of: str | None = None,
) -> dict:
    started = time.monotonic()
    use_cache = settings()["cache"] if use_cache is None else use_cache
    routed = pack_list or route_packs(question)
    risk = risk_of(routed)
    key = cache_key(question, routed, judge, as_of)
    refreshed = {}

    if use_cache and risk != "high":
        result, status = lookup(key)
        if result:
            stale = stale_sources(_doc_ids(result))
            if not stale:
                result["cache_hit"] = True
                result["cache"] = status
                result["refreshed"] = {}
                result["timings"] = {"total_ms": (time.monotonic() - started) * 1000}
                log_latency(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "cache_hit": True,
                        "cache": status,
                        "risk": risk,
                        "index_version": result["index_version"],
                        **result["timings"],
                    }
                )
                return result
            refreshed = stale
            refresh(stale)

    found, corrected, timings, from_evidence = find_evidence(
        question, routed, as_of, reranker, risk, use_cache
    )
    stale = stale_sources({c["doc_id"] for c in found["chunks"]})
    if stale:
        refreshed.update(stale)
        refresh(stale)
        found, corrected, timings, from_evidence = find_evidence(
            question, routed, as_of, reranker, risk, use_cache
        )

    if found["weak"] or not found["ids"]:
        answer, refused = REFUSAL, True
        timings["generate_ms"] = 0.0
    else:
        t0 = time.monotonic()
        answer = chat(
            [
                {"role": "system", "content": system_prompt(found)},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nExcerpts:\n\n{context(found)}",
                },
            ]
        )
        timings["generate_ms"] = (time.monotonic() - t0) * 1000
        refused = False

    t0 = time.monotonic()
    retrieved = {c["id"]: c["text"] for c in found["chunks"]}
    verification = check(answer, retrieved, judge=judge)
    timings["verify_ms"] = (time.monotonic() - t0) * 1000
    timings["total_ms"] = (time.monotonic() - started) * 1000

    by_id = {c["id"]: c for c in found["chunks"]}
    used = [i for i in dict.fromkeys(cited_ids(answer)) if i in by_id]
    if not use_cache:
        cache = "off"
    elif risk == "high":
        cache = "evidence" if from_evidence else "answer not cached (high risk)"
    else:
        cache = "miss"
    if refreshed:
        cache = "source changed: re-ingested"
    result = {
        "question": question,
        "answer": answer,
        "refused": refused,
        "packs": found["packs"],
        "filters": found["filters"],
        "sources": found["ids"],
        "cited": used,
        "citations": [
            {
                "id": i,
                "label": citation_label(by_id[i]),
                "pack": by_id[i]["pack"],
                "doc_id": by_id[i]["doc_id"],
                "version": by_id[i]["version"],
            }
            for i in used
        ],
        "evidence": [
            {
                "claim": c["text"],
                "ids": c["ids"],
                "status": c["status"],
                "passage": c["evidence"],
            }
            for c in verification["claims"]
        ],
        "verification": {
            "counts": verification["counts"],
            "citation_accuracy": verification["citation_accuracy"],
        },
        "strength": found["strength"],
        "corrected": corrected,
        "index_version": current_version(),
        "as_of": as_of,
        "risk": risk,
        "cache": cache,
        "refreshed": refreshed,
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "cache_hit": False,
    }
    log_latency(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "cache_hit": False,
            "cache": cache,
            "risk": risk,
            "index_version": result["index_version"],
            "packs": found["packs"],
            **result["timings"],
        }
    )
    if use_cache:
        save_answer(key, result, risk, live_chunks())
    return result


def _doc_ids(result: dict) -> set:
    return {c["doc_id"] for c in result["citations"] if "doc_id" in c}


def format_result(r: dict) -> str:
    as_of = f"   as of {r['as_of']}" if r.get("as_of") else ""
    lines = [
        f"Packs   : {', '.join(r['packs'])}   risk {r.get('risk', '-')}{as_of}",
        f"Index   : {r.get('index_version', '-')}   cache: {r.get('cache', '-')}",
        "",
        r["answer"],
        "",
    ]
    if r["citations"]:
        lines.append("Citations:")
        lines += [f"  [{c['id']}] {c['label']}" for c in r["citations"]]
    if r["evidence"]:
        lines.append("\nSupporting passages:")
        for e in r["evidence"]:
            lines.append(f"  {e['status']:9} {e['claim'][:90]}")
            if e["passage"]:
                lines.append(f"            > {e['passage'][:160]}")
    v = r["verification"]["counts"]
    lines.append(
        f"\nVerification: {v['supported']} supported, {v['unclear']} unclear, "
        f"{v['wrong']} wrong   evidence strength {r['strength']}"
    )
    if r.get("corrected"):
        c = r["corrected"]
        lines.append(
            f"Self-correction: weak evidence ({c['strength_before']}), "
            f'retried as "{c["query"][:60]}" ({c["strength_after"]})'
        )
    if r.get("refreshed"):
        lines.append(
            f"Freshness: source changed since ingest, re-ingested {r['refreshed']}"
        )
    lines.append(
        "Latency : "
        + "  ".join(f"{k[:-3]} {v:.0f} ms" for k, v in r["timings"].items())
    )
    return "\n".join(lines)


def main():
    utf8_stdout()
    args = sys.argv[1:]
    options = {}
    for name in ("--pack", "--as-of"):
        if name in args:
            i = args.index(name)
            options[name] = args[i + 1]
            args = args[:i] + args[i + 2 :]
    flags = {a for a in args if a.startswith("--")}
    question = " ".join(a for a in args if not a.startswith("--"))
    if not question:
        raise SystemExit(__doc__)
    pack_list = options["--pack"].split(",") if "--pack" in options else None
    r = ask(
        question,
        pack_list,
        use_cache=False if "--no-cache" in flags else None,
        judge="--judge" in flags,
        as_of=options.get("--as-of"),
    )
    print(format_result(r))


if __name__ == "__main__":
    main()
