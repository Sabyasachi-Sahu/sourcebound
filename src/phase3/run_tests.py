"""Phase 3: run a pack's test questions several times and record the score."""

import json
import shutil
import statistics
import sys
import time
from datetime import datetime

import yaml

from src.common.config import PROJECT_ROOT
from src.phase2.check_claims import load_records
from src.phase3.pack_rag import answer
from src.phase3.packs import load_pack
from src.phase3.scoring import HOSPITAL_TYPES, score_question

SCORES_DIR = PROJECT_ROOT / "tests" / "scores"


def parse_args(argv: list) -> dict:
    opts = {"pack": None, "runs": 3, "baseline": False, "only": None}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--runs":
            opts["runs"] = int(argv[i + 1])
            i += 1
        elif arg == "--only":
            opts["only"] = argv[i + 1]
            i += 1
        elif arg == "--baseline":
            opts["baseline"] = True
        elif not arg.startswith("--"):
            opts["pack"] = arg
        i += 1
    if not opts["pack"]:
        raise SystemExit(
            "Usage: python -m src.phase3.run_tests <pack> "
            "[--runs N] [--only ID] [--baseline]"
        )
    return opts


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    opts = parse_args(sys.argv[1:])
    pack = load_pack(opts["pack"])
    spec = yaml.safe_load(pack.test_file.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if opts["only"]:
        questions = [q for q in questions if q["id"] == opts["only"]]
    records = None
    if any(q["type"] in HOSPITAL_TYPES for q in questions):
        records = load_records()

    runs = opts["runs"]
    print(f"Pack: {pack.name}   questions: {len(questions)}   runs: {runs}\n")
    outcomes = {q["id"]: [] for q in questions}
    saved = []
    run_scores = []
    for run in range(1, runs + 1):
        passed_this_run = 0
        for q in questions:
            started = time.time()
            try:
                result = answer(pack, q["question"])
                scored = score_question(pack, q, result, records)
                text = result["answer"]
            except Exception as exc:  # noqa: BLE001 - one failed question must not stop the test run
                error = f"error: {type(exc).__name__}: {exc}"
                scored = {"passed": False, "reasons": [error]}
                text = ""
            scored["seconds"] = round(time.time() - started, 1)
            outcomes[q["id"]].append(scored["passed"])
            passed_this_run += int(scored["passed"])
            saved.append({"run": run, "id": q["id"], "answer": text[:1500], **scored})
            flag = "PASS" if scored["passed"] else "FAIL"
            why = "" if scored["passed"] else "  <- " + "; ".join(scored["reasons"])
            print(f"run {run}  {q['id']:4} {q['type']:13} {flag}{why}")
        run_scores.append(passed_this_run / len(questions))
        print(f"run {run} score: {passed_this_run}/{len(questions)}\n")

    print("Per question (passes out of runs):")
    for q in questions:
        print(f"  {q['id']:4} {q['type']:13} {sum(outcomes[q['id']])}/{runs}")
    mean = statistics.mean(run_scores)
    spread = statistics.pstdev(run_scores) if len(run_scores) > 1 else 0.0
    print(
        f"\nMean score: {mean:.2f}   spread (std dev): {spread:.2f}   "
        f"min {min(run_scores):.2f}  max {max(run_scores):.2f}"
    )

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = SCORES_DIR / f"{pack.name}_{stamp}.json"
    summary = {
        "pack": pack.name,
        "timestamp": stamp,
        "runs": runs,
        "questions": len(questions),
        "run_scores": run_scores,
        "mean": mean,
        "spread": spread,
        "per_question": {k: sum(v) for k, v in outcomes.items()},
        "details": saved,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    if opts["baseline"]:
        baseline = PROJECT_ROOT / "tests" / f"baseline_{pack.name}.json"
        shutil.copyfile(out, baseline)
        print(f"Baseline saved: {baseline}")


if __name__ == "__main__":
    main()
