"""Phase 5: generate the synthetic source documents for a pack with the chat model.

Reads domain_packs/<pack>/docs/briefs.yaml and writes one markdown file per
document. Every listed fact must appear word for word; a section is retried up
to twice when one is missing, and as a last resort the fact is appended as a
"Key rule" line, so the test questions always have their answer in the text.
Existing files are kept (your review is not overwritten) unless --force.

  python -m scripts.gen_pack_docs --pack banking [--force] [--doc CG-POL-001]
"""

import re
import sys
from datetime import date

import yaml

from core.rag.common import chat, chat_model, normalise, numbers, utf8_stdout
from src.common.config import PROJECT_ROOT

PACKS = ["banking", "insurance", "hospital", "pharma"]
RETRIES = 2
SYSTEM = """You write one section of an internal document for a fictional company.
Write 120 to 220 words of plain markdown: paragraphs, numbered steps or bullet
lists. Do not write a heading. Include every listed statement word for word.
Do not add any other numbers, amounts, percentages, time limits or dates.
Do not mention real companies, laws or regulators. Vary sentence length."""


def squash(text: str) -> str:
    return " ".join(normalise(text).lower().replace("*", "").split())


def missing_facts(text: str, facts: list) -> list:
    body = squash(text)
    return [f for f in facts if squash(f) not in body]


def write_section(company: str, doc: dict, section: dict) -> tuple:
    facts = section.get("facts", [])
    prompt = (
        f"Company: {company}\nDocument: {doc['title']} ({doc['doc_type']}, "
        f"version {doc['version']})\nStyle: {doc['style']}\n"
        f"Section heading: {section['heading']}\nWhat it covers: {section['brief']}\n"
    )
    if facts:
        prompt += "Include these statements word for word:\n" + "\n".join(
            f"- {f}" for f in facts
        )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text, tries, gone = "", 0, list(facts)
    for tries in range(1, RETRIES + 2):
        text = chat(messages, temperature=0.7).strip()
        text = re.sub(r"^#+ .*\n", "", text).strip()
        gone = missing_facts(text, facts)
        if not gone:
            return text, tries, []
        messages += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "These statements are missing or changed. "
                "Rewrite the section and include them word for word:\n"
                + "\n".join(f"- {f}" for f in gone),
            },
        ]
    text += "\n\n" + "\n".join(f"Key rule: {f}." for f in gone)
    return text, tries, gone


def extra_numbers(text: str, facts: list) -> set:
    allowed = set().union(*(numbers(f) for f in facts)) if facts else set()
    return {n for n in numbers(text) - allowed if not re.fullmatch(r"[1-9]", n)}


def generate(pack: str, force: bool, only: str | None) -> list:
    folder = PROJECT_ROOT / "domain_packs" / pack / "docs"
    briefs = yaml.safe_load((folder / "briefs.yaml").read_text(encoding="utf-8"))
    report = []
    for doc in briefs["documents"]:
        if only and doc["doc_id"] != only:
            continue
        target = folder / f"{doc['doc_id']}.md"
        if target.exists() and not force:
            report.append(
                (doc["doc_id"], "kept (exists; use --force to replace)", "", "")
            )
            continue
        out = [
            "---",
            f"doc_id: {doc['doc_id']}",
            f'title: "{doc["title"]}"',
            f"doc_type: {doc['doc_type']}",
            f'version: "{doc["version"]}"',
            f'effective_date: "{doc["effective_date"]}"',
            f"owner: {doc['owner']}",
            "synthetic: true",
            "reviewed: false",
            f"generated_by: {chat_model()}",
            f'generated_on: "{date.today()}"',
            "---",
            f"# {doc['title']}",
            "",
            (
                f"> Synthetic document for {briefs['company']}. Not a real policy, "
                "regulation or clinical guidance."
            ),
            "",
        ]
        retried, patched, extras = 0, [], set()
        for section in doc["sections"]:
            text, tries, gone = write_section(briefs["company"], doc, section)
            retried += tries - 1
            patched += gone
            extras |= extra_numbers(text, section.get("facts", []))
            out += [f"## {section['heading']}", "", text, ""]
        target.write_text("\n".join(out), encoding="utf-8")
        note = f"patched {len(patched)}" if patched else "all facts present"
        report.append(
            (
                doc["doc_id"],
                f"written ({note})",
                f"retries {retried}",
                f"other numbers: {sorted(extras)[:6]}" if extras else "",
            )
        )
    return report


def main():
    utf8_stdout()
    args = sys.argv[1:]
    chosen = [args[args.index("--pack") + 1]] if "--pack" in args else PACKS
    only = args[args.index("--doc") + 1] if "--doc" in args else None
    for pack in chosen:
        if pack not in PACKS:
            raise SystemExit(
                f"Unknown pack {pack}. DataSteward uses gen_datasteward_docs."
            )
        print(f"--- {pack} ---")
        for doc_id, status, retries, extras in generate(pack, "--force" in args, only):
            print(f"{doc_id:12} {status:38} {retries:10} {extras}")
    print("\nNext: read each new file, fix anything odd, and set reviewed: true.")


if __name__ == "__main__":
    main()
