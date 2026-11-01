"""Phase 5, Step 5.12: freshness and safe cache invalidation.

The cache never decides by itself that an answer is still true. Every cached
answer stores the chunks it cited and a hash of each chunk's text. On every
read, and in a sweep after every ingest:

  every cited chunk unchanged      -> valid, served
  a cited chunk's text changed     -> medium risk: re-check the answer's claims
                                      against the new text; keep it only if every
                                      claim is still supported
                                      low risk: dropped
  a cited chunk no longer exists   -> dropped
  older than cache_ttl_hours       -> expired (a safety net, not the mechanism)

High-risk packs never cache final answers, only retrieved evidence, and that
evidence cache is keyed by the index version, so any change makes it miss.
Before an answer is trusted, the source files of the chunks behind it are
re-hashed; a file edited since the last ingest is re-ingested first.

  python -m core.rag.freshness status     current version, cache contents, sources
  python -m core.rag.freshness sweep      validate every cached answer now
  python -m core.rag.freshness versions   list index snapshots
  python -m core.rag.freshness report     cache outcomes from the latency log
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta

from core.rag.common import packs, path, read_jsonl, settings, utf8_stdout
from core.rag.index import (
    current_version,
    file_hash,
    ingest,
    read_manifest,
    store_file,
)
from core.rag.verify import check
from src.common.config import PROJECT_ROOT

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def risk_of(pack_names: list) -> str:
    """The strictest risk tier among the packs a question uses."""
    tiers = [packs()[p].get("risk", "high") for p in pack_names]
    return max(tiers, key=RISK_ORDER.get, default="high")


def db() -> sqlite3.Connection:
    file = path("store_dir") / "cache.sqlite"
    file.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(file)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS answer_cache (key TEXT PRIMARY KEY, created TEXT,
            index_version TEXT, risk TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS deps (key TEXT, chunk_id TEXT, chunk_hash TEXT);
        CREATE INDEX IF NOT EXISTS ix_deps_key ON deps (key);
        CREATE TABLE IF NOT EXISTS evidence (key TEXT PRIMARY KEY, created TEXT,
            index_version TEXT, result TEXT);
    """)
    return con


def live_chunks() -> dict:
    """chunk id -> chunk (with its text hash) across every ingested pack."""
    out = {}
    for pack in packs():
        for c in read_jsonl(store_file(pack)):
            out[c["id"]] = c
    return out


# ---------- answers ----------


def save_answer(key: str, result: dict, risk: str, chunks: dict):
    """Cache an answer with the hash of every chunk it cited."""
    if risk == "high" or result["refused"] or not result["cited"]:
        return False
    with db() as con:
        con.execute("DELETE FROM deps WHERE key = ?", [key])
        con.execute(
            "INSERT OR REPLACE INTO answer_cache VALUES (?, ?, ?, ?, ?)",
            [
                key,
                datetime.now().isoformat(timespec="seconds"),
                result["index_version"],
                risk,
                json.dumps(result),
            ],
        )
        con.executemany(
            "INSERT INTO deps VALUES (?, ?, ?)",
            [(key, cid, chunks[cid]["hash"]) for cid in result["cited"]],
        )
    return True


def _validate(con, key: str, created: str, risk: str, result: dict, live: dict) -> str:
    """Decide what to do with one cached answer. Returns its new status."""
    ttl = timedelta(hours=settings()["cache_ttl_hours"])
    if datetime.now() - datetime.fromisoformat(created) > ttl:
        _drop(con, key)
        return "expired"
    deps = con.execute(
        "SELECT chunk_id, chunk_hash FROM deps WHERE key = ?", [key]
    ).fetchall()
    changed = [cid for cid, h in deps if cid in live and live[cid]["hash"] != h]
    gone = [cid for cid, _ in deps if cid not in live]
    if not changed and not gone:
        return "valid"
    if gone or risk != "medium":
        _drop(con, key)
        return "dropped"
    texts = {cid: live[cid]["text"] for cid, _ in deps}
    verdict = check(result["answer"], texts)
    counts = verdict["counts"]
    if counts["wrong"] or counts["unclear"] or not counts["supported"]:
        _drop(con, key)
        return "dropped"
    result["evidence"] = [
        {
            "claim": c["text"],
            "ids": c["ids"],
            "status": c["status"],
            "passage": c["evidence"],
        }
        for c in verdict["claims"]
    ]
    result["rechecked"] = datetime.now().isoformat(timespec="seconds")
    con.execute(
        "UPDATE answer_cache SET result = ? WHERE key = ?", [json.dumps(result), key]
    )
    con.execute("DELETE FROM deps WHERE key = ?", [key])
    con.executemany(
        "INSERT INTO deps VALUES (?, ?, ?)",
        [(key, cid, live[cid]["hash"]) for cid, _ in deps],
    )
    return "rechecked"


def _drop(con, key: str):
    con.execute("DELETE FROM answer_cache WHERE key = ?", [key])
    con.execute("DELETE FROM deps WHERE key = ?", [key])


def lookup(key: str) -> tuple:
    """Return (result or None, status). Validates the entry before serving it."""
    with db() as con:
        row = con.execute(
            "SELECT created, risk, result FROM answer_cache WHERE key = ?", [key]
        ).fetchone()
        if not row:
            return None, "miss"
        result = json.loads(row[2])
        status = _validate(con, key, row[0], row[1], result, live_chunks())
    if status in ("valid", "rechecked"):
        return result, status
    return None, status


def sweep() -> dict:
    """Validate every cached answer, e.g. right after an ingest."""
    counts = {"valid": 0, "rechecked": 0, "dropped": 0, "expired": 0}
    live = live_chunks()
    with db() as con:
        rows = con.execute(
            "SELECT key, created, risk, result FROM answer_cache"
        ).fetchall()
        for key, created, risk, result in rows:
            counts[_validate(con, key, created, risk, json.loads(result), live)] += 1
        version = current_version()
        con.execute("DELETE FROM evidence WHERE index_version != ?", [version])
    log = path("store_dir") / "freshness_log.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "index_version": current_version(),
                    **counts,
                }
            )
            + "\n"
        )
    return counts


# ---------- evidence (high risk) ----------


def get_evidence(key: str):
    with db() as con:
        row = con.execute("SELECT result FROM evidence WHERE key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else None


def save_evidence(key: str, found: dict):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)",
            [
                key,
                datetime.now().isoformat(timespec="seconds"),
                current_version(),
                json.dumps(found),
            ],
        )


# ---------- source freshness ----------


def stale_sources(doc_ids: set) -> dict:
    """pack -> [doc_id] for documents whose source file changed since ingest."""
    manifest = read_manifest()
    stale, hashed = {}, {}
    for pack, docs in manifest.items():
        for doc_id in doc_ids & docs.keys():
            entry = docs[doc_id]
            origin = entry.get("origin")
            if not origin:
                continue
            file = PROJECT_ROOT / origin
            if origin not in hashed:
                hashed[origin] = file_hash(file) if file.exists() else "missing"
            if hashed[origin] != entry.get("file_hash"):
                stale.setdefault(pack, []).append(doc_id)
    return stale


def refresh(stale: dict) -> str:
    """Re-ingest the packs with changed sources, then sweep the cache."""
    for pack in stale:
        ingest(pack)
    sweep()
    return current_version()


# ---------- commands ----------


def status():
    manifest = read_manifest()
    all_docs = {d for docs in manifest.values() for d in docs}
    stale = stale_sources(all_docs)
    with db() as con:
        answers = con.execute(
            "SELECT risk, COUNT(*) FROM answer_cache GROUP BY risk"
        ).fetchall()
        evidence = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    print(f"Index version     : {current_version()}")
    print(f"Cached answers    : {dict(answers) or 0}")
    print(f"Cached evidence   : {evidence} (high-risk packs)")
    tiers = {p: spec.get("risk", "high") for p, spec in packs().items()}
    print(f"Risk tiers        : {tiers}")
    if stale:
        print(f"STALE SOURCES     : {stale}")
        print("                    edited since the last ingest; run index ingest")
    else:
        print("Sources           : every source file matches the index")


def versions():
    folder = path("snapshot_dir")
    files = sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime)
    current = current_version()
    for file in files:
        snap = json.loads(file.read_text(encoding="utf-8"))
        docs = sum(len(d) for d in snap["manifest"].values())
        mark = "  <- current" if snap["index_version"] == current else ""
        print(f"{snap['index_version']}  {snap['created']}  {docs:>5} documents{mark}")


def report():
    rows = read_jsonl(path("store_dir") / "latency.jsonl")
    outcomes = {}
    for r in rows:
        outcomes[r.get("cache", "-")] = outcomes.get(r.get("cache", "-"), 0) + 1
    print(f"Questions logged : {len(rows)}")
    for name, n in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {name:12} {n}")
    sweeps = read_jsonl(path("store_dir") / "freshness_log.jsonl")
    for s in sweeps[-5:]:
        print(
            f"Sweep {s['time']}  index {s['index_version']}  valid {s['valid']}  "
            f"re-checked {s['rechecked']}  dropped {s['dropped']}  "
            f"expired {s['expired']}"
        )


def main():
    utf8_stdout()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "status":
        status()
    elif command == "sweep":
        print(sweep())
    elif command == "versions":
        versions()
    elif command == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
