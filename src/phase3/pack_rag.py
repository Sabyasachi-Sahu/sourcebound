"""Phase 3: grounded answer for any pack, with a pack-aware citation check."""

import sys

import ollama

from src.common.config import get_settings
from src.phase3.pack_search import search
from src.phase3.packs import Pack, load_pack

HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2212"
HYPHEN_TABLE = str.maketrans(HYPHENS, "-" * len(HYPHENS))


def build_context(hits: list) -> str:
    return "\n\n".join(f"[{hit['id']}]\n{hit['text']}" for hit in hits)


def find_ids(pack: Pack, text: str) -> set:
    cleaned = text.replace("\\", "").translate(HYPHEN_TABLE).lower()
    return set(pack.id_pattern.findall(cleaned))


def check_citations(pack: Pack, text: str, sources: list) -> dict:
    cited = find_ids(pack, text)
    return {
        "verified": sorted(cited & set(sources)),
        "unverified": sorted(cited - set(sources)),
    }


def answer(pack: Pack, question: str) -> dict:
    s = get_settings()
    hits = search(pack, question)
    context = build_context(hits)
    response = ollama.chat(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": pack.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {
        "answer": response.message.content,
        "sources": [h["id"] for h in hits],
        "hits": hits,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python -m src.phase3.pack_rag <pack> "<question>"')
    pack = load_pack(sys.argv[1])
    question = " ".join(sys.argv[2:])
    result = answer(pack, question)
    print(f"Pack: {pack.name}\nQuestion: {question}\n")
    print(result["answer"])
    print("\nRetrieved sources:")
    for source_id in result["sources"]:
        print(f"  - {source_id}")
    check = check_citations(pack, result["answer"], result["sources"])
    print(
        f"\nCitation check: {len(check['verified'])} verified, "
        f"{len(check['unverified'])} unverified"
    )
    for bad in check["unverified"]:
        print(f"  WARNING not retrieved: {bad}")


if __name__ == "__main__":
    main()
