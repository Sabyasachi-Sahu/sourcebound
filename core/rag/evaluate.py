"""Phase 5: validate the RAG test sets, measure them, and compare with Phase 3.

validate                           check every gold answer exists in the documents
run --pack <pack|all|cross> [--runs N] [--baseline] [--no-rerank] [--judge]
compare --pack <hospital|pharma> [--runs N]   Phase 3 engine vs Phase 5 engine
calibrate                          evidence strength of answerable vs refuse questions
latency                            per-stage latency from the log
report                             print the saved baselines
"""

import json
import statistics
import sys
from datetime import datetime

import yaml

from core.rag.answer import REFUSAL, ask
from core.rag.common import packs, path, read_jsonl, utf8_stdout
from core.rag.index import store_file
from core.rag.retrieve import choose_scope, retrieve, route_packs
from src.common.config import PROJECT_ROOT

TESTS = PROJECT_ROOT / "tests" / "rag"
SCORES = PROJECT_ROOT / "tests" / "scores"
BASELINE = PROJECT_ROOT / "tests" / "baseline_rag.json"
SETS = [*packs(), "cross"]
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain",
    "no information",
    "not provided",
    "not include",
    "not available",
    "cannot",
    "can't",
    "unable",
    "not mentioned",
    "not covered",
    "not enough information",
)


def fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def squash(text: str) -> str:
    return " ".join(text.lower().replace("*", "").split())


def questions(name: str) -> list:
    file = TESTS / f"{name}_p5.yaml"
    if not file.exists():
        raise SystemExit(f"Test file not found: {file}")
    return yaml.safe_load(file.read_text(encoding="utf-8"))["questions"]


def chunks_by_doc() -> dict:
    out = {}
    for pack in packs():
        for c in read_jsonl(store_file(pack)):
            out.setdefault(c["doc_id"], []).append(c)
    return out


def validate() -> int:
    docs, bad = chunks_by_doc(), 0
    for name in SETS:
        print(f"--- {name} ---")
        for q in questions(name):
            status, note = "OK", ""
            if q["type"] == "doc_fact":
                doc = docs.get(q["expect_doc"])
                if not doc:
                    status, note = "FAIL", f"document {q['expect_doc']} is not ingested"
                else:
                    text = squash(" ".join(c["text"] for c in doc))
                    gone = [t for t in q["expect_terms"] if squash(t) not in text]
                    scope = choose_scope(doc[0]["pack"], q["question"])
                    if gone:
                        status, note = "FAIL", f"terms not in the document: {gone}"
                    elif scope is not None and scope != doc[0]["scope"]:
                        status, note = (
                            "FAIL",
                            (
                                f"question goes to scope '{scope}', the "
                                f"document is in '{doc[0]['scope']}'"
                            ),
                        )
            elif q["type"] == "cross":
                routed = route_packs(q["question"])
                gone = [p for p in q["expect_packs"] if p not in routed]
                if gone:
                    status, note = (
                        "FAIL",
                        f"router does not pick {gone} (picks {routed})",
                    )
            bad += status != "OK"
            print(f"{q['id']:5} {q['type']:9} {status:4} {note}")
    print("--- Phase 3 sets (must stay in the Phase 3 scope) ---")
    for name in ("hospital", "pharma"):
        file = PROJECT_ROOT / "tests" / f"{name}_p3.yaml"
        if not file.exists():
            continue
        default = packs()[name]["default_scope"]
        for q in yaml.safe_load(file.read_text(encoding="utf-8"))["questions"]:
            scope = choose_scope(name, q["question"])
            ok = scope == default
            bad += not ok
            note = "" if ok else f"goes to '{scope}', expected '{default}'"
            print(f"{q['id']:5} {name:9} {'OK' if ok else 'FAIL':4} {note}")
    print(
        f"\n{bad} question(s) need attention"
        if bad
        else "\nAll gold answers are in the documents and every question routes"
    )
    return bad


def refused(answer: str) -> bool:
    text = answer.lower()
    return answer.startswith(REFUSAL[:40]) or any(m in text for m in REFUSAL_MARKERS)


def score(q: dict, r: dict) -> dict:
    answer = squash(r["answer"])
    by_id = {}
    for pack in r["packs"]:
        for c in read_jsonl(store_file(pack)):
            by_id[c["id"]] = c
    reasons, hit = [], None
    if q["type"] == "refuse":
        if not refused(r["answer"]):
            reasons.append("did not refuse")
    else:
        if r["refused"]:
            reasons.append("refused an answerable question")
        gone = [t for t in q["expect_terms"] if squash(t) not in answer]
        if gone:
            reasons.append(f"missing {gone}")
        cited = [by_id[c["id"]] for c in r["citations"] if c["id"] in by_id]
        if q["type"] == "doc_fact":
            want = q["expect_doc"]
            hit = any(by_id.get(s, {}).get("doc_id") == want for s in r["sources"])
            if not any(c["doc_id"] == want for c in cited):
                reasons.append(f"does not cite {want}")
        else:
            hit = all(
                any(by_id.get(s, {}).get("pack") == p for s in r["sources"])
                for p in q["expect_packs"]
            )
            gone = [p for p in q["expect_packs"] if p not in {c["pack"] for c in cited}]
            if gone:
                reasons.append(f"no citation from {gone}")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "hit": hit,
        "citation_accuracy": r["verification"]["citation_accuracy"],
        "refused": r["refused"] or refused(r["answer"]),
        "ms": r["timings"]["total_ms"],
    }


def summarise(records: list, runs: int) -> dict:
    per_run = [
        statistics.mean(r["passed"] for r in records if r["run"] == n)
        for n in range(1, runs + 1)
    ]
    hits = [r["hit"] for r in records if r["hit"] is not None]
    acc = [
        r["citation_accuracy"] for r in records if r["citation_accuracy"] is not None
    ]
    refuse = [r for r in records if r["type"] == "refuse"]
    answerable = [r for r in records if r["type"] != "refuse"]
    ms = sorted(r["ms"] for r in records)
    return {
        "runs": runs,
        "questions": len(records) // runs,
        "run_scores": per_run,
        "mean": round(statistics.mean(per_run), 4),
        "spread": round(statistics.pstdev(per_run), 4) if runs > 1 else 0.0,
        "hit_rate": round(statistics.mean(hits), 4) if hits else None,
        "citation_accuracy": round(statistics.mean(acc), 4) if acc else None,
        "refusal_rate": (
            round(statistics.mean(r["refused"] for r in refuse), 4) if refuse else None
        ),
        "false_refusals": sum(r["refused"] for r in answerable),
        "latency_p50_ms": round(ms[len(ms) // 2]),
        "latency_p95_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))]),
    }


def save_baseline(key: str, value: dict):
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    base[key] = value
    BASELINE.write_text(json.dumps(base, indent=2), encoding="utf-8")


def run(name: str, runs: int, save: bool, reranker: str | None, judge: bool) -> dict:
    qs = questions(name)
    records = []
    for n in range(1, runs + 1):
        for q in qs:
            pack_list = None if name == "cross" else [name]
            try:
                r = ask(
                    q["question"],
                    pack_list,
                    use_cache=False,
                    judge=judge,
                    reranker=reranker,
                )
                s = score(q, r)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
                s = {
                    "passed": False,
                    "reasons": [f"error: {type(exc).__name__}: {exc}"],
                    "hit": None,
                    "citation_accuracy": None,
                    "refused": False,
                    "ms": 0,
                }
                r = {"answer": "", "cited": []}
            mark = "PASS" if s["passed"] else "FAIL"
            why = f"  <- {'; '.join(s['reasons'])}" if s["reasons"] else ""
            print(f"run {n}  {q['id']:5} {q['type']:9} {mark}{why[:120]}")
            records.append(
                {
                    "run": n,
                    "id": q["id"],
                    "type": q["type"],
                    **s,
                    "answer": r["answer"],
                    "cited": r["cited"],
                }
            )
    summary = summarise(records, runs)
    variant = name + ("_norerank" if reranker == "none" else "")
    print(
        f"\n{variant}: pass {summary['mean']:.2f} (spread {summary['spread']:.2f})  "
        f"hit rate {fmt(summary['hit_rate'])}  "
        f"citation accuracy {fmt(summary['citation_accuracy'])}  "
        f"refusal rate {fmt(summary['refusal_rate'])}  "
        f"false refusals {summary['false_refusals']}  "
        f"latency p50 {summary['latency_p50_ms']} ms p95 {summary['latency_p95_ms']} ms"
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SCORES.mkdir(parents=True, exist_ok=True)
    detail = SCORES / f"rag_{variant}_{stamp}.json"
    detail.write_text(
        json.dumps({**summary, "records": records}, indent=1), encoding="utf-8"
    )
    print(f"Saved: {detail}")
    if save:
        save_baseline(variant, {**summary, "created": stamp})
        print(f"Baseline saved for '{variant}'")
    return summary


def compare(pack_name: str, runs: int):
    """Run the Phase 3 test set through the Phase 3 and Phase 5 engines, same scorer."""
    from src.phase3.pack_rag import answer as p3_answer
    from src.phase3.packs import load_pack
    from src.phase3.scoring import score_question

    pack = load_pack(pack_name)
    tests = yaml.safe_load(
        (PROJECT_ROOT / "tests" / f"{pack_name}_p3.yaml").read_text(encoding="utf-8")
    )["questions"]
    # Each engine is scored against the excerpt texts it actually showed the model.
    texts = {
        "p3": {r["id"]: r["text"] for r in read_jsonl(pack.docs_file)},
        "p5": {c["id"]: c["text"] for c in read_jsonl(store_file(pack_name))},
    }
    results = {"p3": [], "p5": []}
    for n in range(1, runs + 1):
        for q in tests:
            for engine in ("p3", "p5"):
                try:
                    if engine == "p3":
                        r = p3_answer(pack, q["question"])
                    else:
                        out = ask(q["question"], [pack_name], use_cache=False)
                        r = {"answer": out["answer"], "sources": out["sources"]}
                    s = score_question(pack, q, r, texts[engine])
                except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
                    s = {"passed": False, "reasons": [f"error: {type(exc).__name__}"]}
                results[engine].append({"run": n, "id": q["id"], "passed": s["passed"]})
                mark = "PASS" if s["passed"] else "FAIL"
                print(
                    f"run {n}  {q['id']:5} {engine}  {mark}  "
                    f"{'; '.join(s['reasons'])[:90]}"
                )
    print(f"\n{'question':9} {'Phase 3':>8} {'Phase 5':>8}")
    for q in tests:
        a = sum(r["passed"] for r in results["p3"] if r["id"] == q["id"])
        b = sum(r["passed"] for r in results["p5"] if r["id"] == q["id"])
        print(f"{q['id']:9} {a:>6}/{runs} {b:>6}/{runs}")
    summary = {}
    for engine, rows in results.items():
        per_run = [
            statistics.mean(r["passed"] for r in rows if r["run"] == n)
            for n in range(1, runs + 1)
        ]
        summary[engine] = {
            "mean": round(statistics.mean(per_run), 4),
            "spread": round(statistics.pstdev(per_run), 4) if runs > 1 else 0.0,
        }
    delta = summary["p5"]["mean"] - summary["p3"]["mean"]
    verdict = (
        "BEATEN"
        if delta > 0
        else "EQUAL AT 1.00"
        if summary["p5"]["mean"] == 1
        else "NOT BEATEN"
    )
    print(
        f"\n{pack_name}: Phase 3 {summary['p3']['mean']:.2f} (spread "
        f"{summary['p3']['spread']:.2f})  Phase 5 {summary['p5']['mean']:.2f} (spread "
        f"{summary['p5']['spread']:.2f})  change {delta:+.2f}  {verdict}"
    )
    save_baseline(
        f"compare_{pack_name}",
        {
            **summary,
            "runs": runs,
            "verdict": verdict,
            "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
    )
    print(f"Saved to {BASELINE}")


def calibrate():
    """Evidence strength for answerable and refuse questions, to set the thresholds."""
    rows = []
    for name in packs():
        for q in questions(name):
            for reranker in ("llm", "none"):
                r = retrieve(q["question"], [name], reranker=reranker)
                best_sim = max(r["similarity"].values(), default=0)
                best_rr = max(r["scores"].values()) if r["scores"] else None
                rows.append((q["id"], q["type"], reranker, best_sim, best_rr))
    print(f"{'id':6} {'type':9} {'best similarity':>15} {'best rerank':>11}")
    for qid, kind, reranker, sim, rr in rows:
        if reranker == "llm":
            print(f"{qid:6} {kind:9} {sim:15.3f} {('-' if rr is None else rr):>11}")
    for label, idx, mode in (("similarity", 3, "none"), ("rerank", 4, "llm")):
        good = [r[idx] for r in rows if r[1] != "refuse" and r[2] == mode]
        bad = [r[idx] for r in rows if r[1] == "refuse" and r[2] == mode]
        good = [g for g in good if g is not None]
        bad = [b for b in bad if b is not None]
        if good and bad:
            split = min(good) > max(bad)
            verdict = "a threshold between them separates" if split else "they overlap"
            print(
                f"{label:10}: answerable min {min(good):.3f}, "
                f"refuse max {max(bad):.3f}  -> {verdict}"
            )


def latency():
    rows = read_jsonl(path("store_dir") / "latency.jsonl")
    if not rows:
        print("No latency entries yet.")
        return
    fresh = [r for r in rows if not r["cache_hit"]]
    hits = [r for r in rows if r["cache_hit"]]
    print(f"Questions: {len(rows)}  (cache hits {len(hits)})")
    for stage in (
        "embed_ms",
        "search_ms",
        "rerank_ms",
        "retry_ms",
        "generate_ms",
        "verify_ms",
        "total_ms",
    ):
        values = sorted(r[stage] for r in fresh if stage in r)
        if values:
            p50 = values[len(values) // 2]
            p95 = values[min(len(values) - 1, int(len(values) * 0.95))]
            print(
                f"  {stage[:-3]:9} p50 {p50:>8.0f} ms   "
                f"p95 {p95:>8.0f} ms   n={len(values)}"
            )
    if hits:
        cached = sorted(r["total_ms"] for r in hits)[len(hits) // 2]
        print(f"  cached    p50 {cached:>8.0f} ms")


def report():
    if not BASELINE.exists():
        raise SystemExit("No baseline yet.")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(
        f"{'set':22} {'pass':>5} {'spread':>6} {'hit':>5} {'cite acc':>8} "
        f"{'refusal':>7} {'p50 ms':>7}"
    )
    for key, b in base.items():
        if key.startswith("compare_"):
            print(
                f"{key:22} Phase 3 {b['p3']['mean']:.2f} -> "
                f"Phase 5 {b['p5']['mean']:.2f}  {b['verdict']}"
            )
            continue
        print(
            f"{key:22} {b['mean']:5.2f} {b['spread']:6.2f} {fmt(b['hit_rate']):>5} "
            f"{fmt(b['citation_accuracy']):>8} {fmt(b['refusal_rate']):>7} "
            f"{b['latency_p50_ms']:>7}"
        )


def main():
    utf8_stdout()
    args = sys.argv[1:]
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 3
    if args[:1] == ["validate"]:
        sys.exit(int(validate() > 0))
    elif args[:1] == ["run"] and "--pack" in args:
        target = args[args.index("--pack") + 1]
        reranker = "none" if "--no-rerank" in args else None
        for name in SETS if target == "all" else [target]:
            run(name, runs, "--baseline" in args, reranker, "--judge" in args)
    elif args[:1] == ["compare"] and "--pack" in args:
        compare(args[args.index("--pack") + 1], runs)
    elif args[:1] == ["calibrate"]:
        calibrate()
    elif args[:1] == ["latency"]:
        latency()
    elif args[:1] == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
