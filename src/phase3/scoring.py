"""Phase 3: score one answer against one test question, using code, not reading."""

from src.phase2.check_claims import check_claims, record_conditions, words
from src.phase3.pack_rag import find_ids
from src.phase3.packs import Pack

REFUSAL_MARKERS = (
    "does not contain",
    "do not contain",
    "not contain",
    "no information",
    "not available",
    "not provided",
    "not include",
    "no record",
    "not recorded",
    "cannot",
    "can't",
    "unable",
    "not in the",
    "not enough",
)
SAMPLE_MARKERS = (
    "sample",
    "full database",
    "not complete",
    "cannot",
    "can't",
    "unable",
    "not possible",
    "not enough",
)
HOSPITAL_TYPES = {"find", "comorbidity", "refuse_count", "unanswerable"}


def has_marker(text: str, markers: tuple) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def patient_has(record_text: str, term: str) -> bool:
    wanted = words(term)
    return any(wanted <= condition for condition in record_conditions(record_text))


def result_base(passed: bool, reasons: list, cited: set, unverified: list, **extra):
    return {
        "passed": passed,
        "reasons": reasons,
        "cited": len(cited),
        "unverified": len(unverified),
        **extra,
    }


def score_question(
    pack: Pack, q: dict, result: dict, records: dict | None = None
) -> dict:
    answer = result["answer"]
    # IDs the user typed in the question are not citations
    cited = find_ids(pack, answer) - find_ids(pack, q["question"])
    unverified = sorted(cited - set(result["sources"]))
    reasons = []
    kind = q["type"]

    if unverified:
        reasons.append(f"{len(unverified)} cited ID(s) were not retrieved")

    if kind == "find":
        if len(cited) < q.get("min_cited", 2):
            reasons.append(f"only {len(cited)} patient(s) cited")
        wrong = [
            pid
            for pid in sorted(cited)
            if pid not in records or not patient_has(records[pid], q["term"])
        ]
        if wrong:
            reasons.append(f"{len(wrong)} cited patient(s) lack '{q['term']}'")
        return result_base(not reasons, reasons, cited, unverified)

    if kind == "comorbidity":
        if len(cited) < 2:
            reasons.append(f"only {len(cited)} patient(s) cited")
        claims = check_claims(answer, records)
        counts = {
            s: sum(c["status"] == s for c in claims)
            for s in ("supported", "wrong", "unclear")
        }
        if not claims:
            reasons.append("no checkable claims (no bold condition names with IDs)")
        if counts["wrong"]:
            reasons.append(f"{counts['wrong']} wrong claim(s)")
        if claims and not counts["supported"]:
            reasons.append("no supported claims")
        return result_base(not reasons, reasons, cited, unverified, claims=counts)

    if kind == "refuse_count":
        if not has_marker(answer, SAMPLE_MARKERS):
            reasons.append("answer does not say it only sees a sample")
        return result_base(not reasons, reasons, cited, unverified)

    if kind == "unanswerable":
        if not has_marker(answer, REFUSAL_MARKERS):
            reasons.append("answer does not say the information is missing")
        return result_base(not reasons, reasons, cited, unverified)

    if kind == "label_fact":
        by_id = {h["id"]: h["text"].lower() for h in result.get("hits", [])}
        sections = q.get("sections")
        terms = [t.lower() for t in q["expect_any"]]
        right = []
        for cid in sorted(cited):
            parts = cid.split(".")
            if len(parts) != 3 or parts[0] != q["drug"]:
                continue
            if sections and parts[1] not in sections:
                continue
            right.append(cid)
        if not right:
            where = f" in {sections}" if sections else ""
            reasons.append(f"no cited excerpt from {q['drug']}{where}")
        elif not any(any(t in by_id.get(cid, "") for t in terms) for cid in right):
            reasons.append("cited excerpts do not contain an expected term")
        if not any(t in answer.lower() for t in terms):
            reasons.append("answer does not contain an expected term")
        return result_base(not reasons, reasons, cited, unverified)

    raise ValueError(f"Unknown question type: {kind}")
