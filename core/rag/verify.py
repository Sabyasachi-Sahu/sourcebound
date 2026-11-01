"""Phase 5: check every cited claim against the passage it cites.

Each sentence with a citation is a claim. Outcomes, as in Phase 2:
  supported  most of the claim's words, and all its numbers, are in a cited passage
  wrong      it cites an excerpt that was not retrieved, or shares almost nothing
             with the cited passage, or states a number the passage does not have
  unclear    in between: a paraphrase the rule cannot decide
An optional model judge (--judge) re-checks only the unclear claims.
"""

import re

from core.rag.common import chat, normalise, numbers, tokens

CITE = re.compile(r"\[([^\[\]]+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\"(])")
ID_SPLIT = re.compile(r"\s*[,;]\s*|\s+and\s+")
SUPPORTED, WRONG = 0.6, 0.25
JUDGE_PROMPT = """You judge one claim against one passage. Reply with one word:
SUPPORTED if the passage states the claim, CONTRADICTED if the passage says
something different, NOT_ENOUGH if the passage does not decide it."""


def cited_ids(text: str, known: set | None = None) -> list:
    """IDs inside [ ... ], split on commas. Unknown words in brackets are ignored
    when the set of known IDs is given."""
    found = []
    for group in CITE.findall(normalise(text)):
        for part in ID_SPLIT.split(group.strip()):
            part = part.strip().strip("`*")
            if part and (
                known is None or part in known or re.fullmatch(r"[\w.:-]+", part)
            ):
                found.append(part)
    return found


def claims(answer: str) -> list:
    out = []
    for sentence in SENTENCE.split(normalise(answer).replace("\n", " ")):
        ids = cited_ids(sentence)
        if ids:
            text = CITE.sub(" ", sentence)
            text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(text.split()))
            out.append({"text": text, "ids": ids})
    return out


def best_sentence(claim: str, passage: str) -> str:
    first, _, rest = passage.partition("\n")
    if " | " in first and rest:
        passage = rest
    want = set(tokens(claim))
    parts = [
        p.strip()
        for p in re.split(r"(?<=[.!?])\s+", " ".join(passage.split()))
        if p.strip()
    ]
    return max(parts, key=lambda p: len(want & set(tokens(p))), default="")


def check(answer: str, retrieved: dict, judge: bool = False) -> dict:
    """retrieved: chunk id -> chunk text. Returns per-claim results and counts."""
    results = []
    for claim in claims(answer):
        words = set(tokens(claim["text"])) - {
            t for i in claim["ids"] for t in tokens(i)
        }
        nums = numbers(claim["text"])
        missing = [i for i in claim["ids"] if i not in retrieved]
        cited = [retrieved[i] for i in claim["ids"] if i in retrieved]
        coverage, passage = 0.0, ""
        for text in cited:
            have = set(tokens(text))
            cov = len(words & have) / len(words) if words else 1.0
            if cov >= coverage:
                coverage, passage = cov, text
        nums_ok = all(n in numbers(" ".join(cited)) for n in nums)
        if missing and not cited:
            status, why = "wrong", f"cites {missing[0]}, which was not retrieved"
        elif coverage >= SUPPORTED and nums_ok:
            status, why = "supported", f"coverage {coverage:.2f}"
        elif coverage < WRONG or (nums and not nums_ok and numbers(" ".join(cited))):
            status, why = "wrong", f"coverage {coverage:.2f}, numbers ok: {nums_ok}"
        else:
            status, why = "unclear", f"coverage {coverage:.2f}, numbers ok: {nums_ok}"
        if judge and status == "unclear" and passage:
            verdict = chat(
                [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Claim: {claim['text']}\n\nPassage: {passage}",
                    },
                ]
            ).upper()
            if "SUPPORTED" in verdict:
                status, why = "supported", "judge: supported"
            elif "CONTRADICTED" in verdict:
                status, why = "wrong", "judge: contradicted"
        results.append(
            {
                **claim,
                "status": status,
                "why": why,
                "evidence": best_sentence(claim["text"], passage),
            }
        )
    counts = {
        s: sum(r["status"] == s for r in results)
        for s in ("supported", "unclear", "wrong")
    }
    total = len(results)
    return {
        "claims": results,
        "counts": counts,
        "citation_accuracy": round(counts["supported"] / total, 3) if total else None,
    }
