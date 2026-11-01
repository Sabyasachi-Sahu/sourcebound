"""Phase 3: check that each test question's gold answer exists in the real data."""

import json
import sys

import yaml

from src.common.config import PROJECT_ROOT
from src.phase2.check_claims import load_records
from src.phase3.packs import load_pack
from src.phase3.scoring import patient_has


def validate_hospital(questions: list) -> int:
    records = load_records()
    bad = 0
    print(f"Patient records loaded: {len(records)}\n")
    for q in questions:
        term = q.get("term")
        if not term:
            print(f"{q['id']:4} {q['type']:13} no term to check")
            continue
        n = sum(1 for text in records.values() if patient_has(text, term))
        need = q.get("min_cited", 2)
        ok = n >= need
        bad += int(not ok)
        note = ""
        if q["type"] == "refuse_count":
            note = "  (real count; retrieval returns only the top few)"
        status = "OK" if ok else "FAIL"
        print(f"{q['id']:4} {q['type']:13} '{term}': {n} patients  {status}{note}")
    return bad


def validate_pharma(pack, questions: list) -> int:
    if not pack.docs_file.exists():
        raise SystemExit(f"{pack.docs_file} not found. Run build_pharma_docs first.")
    chunks = [json.loads(line) for line in pack.docs_file.open(encoding="utf-8")]
    bad = 0
    print(f"Chunks loaded: {len(chunks)}\n")
    for q in questions:
        if q["type"] != "label_fact":
            print(f"{q['id']:4} {q['type']:13} nothing to check")
            continue
        drug = q["drug"]
        of_drug = [c for c in chunks if c["metadata"]["drug"] == drug]
        sections = q.get("sections")
        pool = [
            c for c in of_drug if not sections or c["metadata"]["section"] in sections
        ]
        terms = [t.lower() for t in q["expect_any"]]
        hits = [c for c in pool if any(t in c["text"].lower() for t in terms)]
        if not of_drug:
            status, detail = "FAIL", f"no chunks for {drug}"
        elif not pool:
            have = sorted({c["metadata"]["section"] for c in of_drug})
            status, detail = "FAIL", f"none in {sections}; {drug} has {have}"
        elif not hits:
            status, detail = "FAIL", f"{len(pool)} chunks but none contain {terms}"
        else:
            status = "OK"
            detail = f"{len(hits)} of {len(pool)} chunks contain an expected term"
        bad += int(status != "OK")
        print(f"{q['id']:4} {drug:20} {status:5} {detail}")
    return bad


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.phase3.validate_tests <pack>")
    pack = load_pack(sys.argv[1])
    spec = yaml.safe_load(pack.test_file.read_text(encoding="utf-8"))
    questions = spec["questions"]
    print(f"Pack: {pack.name}   test file: {pack.test_file.relative_to(PROJECT_ROOT)}")
    if pack.name == "hospital":
        bad = validate_hospital(questions)
    else:
        bad = validate_pharma(pack, questions)
    if bad:
        print(f"\n{bad} question(s) need attention")
    else:
        print("\nAll checked questions are grounded in the data")


if __name__ == "__main__":
    main()
