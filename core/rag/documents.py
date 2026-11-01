"""Phase 5: load source documents for each pack and split them into chunks.

Three source types (core/rag/rag.yaml):
  markdown_dir     synthetic documents with front matter, in domain_packs/<pack>/docs
  pharma_labels    the Phase 3 openFDA label files, rendered to markdown first
  patient_records  the Phase 2/3 patient documents, one chunk per patient

Chunks never cross a section, keep their file line numbers for citations, and
keep the Phase 3 IDs for labels (drug.section.nn) and patients (patient ID), so
the Phase 3 test sets and scorer still work.
"""

import json
import re
import sys
import textwrap

import yaml

from core.rag.common import packs, path, settings, utf8_stdout
from src.common.config import PROJECT_ROOT
from src.phase3.packs import load_pack

HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
SENTENCE_END = re.compile(r"[.!?:;)\]]\s*$")
FRONT_KEYS = {
    "doc_id",
    "title",
    "doc_type",
    "version",
    "effective_date",
    "synthetic",
    "reviewed",
    "owner",
}


def parse_markdown(
    file, pack: str, scope: str, id_style: str = "generic", extra: dict | None = None
) -> dict:
    lines = file.read_text(encoding="utf-8-sig").splitlines()
    meta, start = {}, 0
    if lines and lines[0].strip() == "---":
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
        start = end + 1
    missing = {"doc_id", "title"} - set(meta)
    if missing:
        raise SystemExit(f"{file}: front matter is missing {sorted(missing)}")
    sections, current, major, minor = [], None, 0, 0
    for number in range(start, len(lines)):
        match = HEADING.match(lines[number])
        if match and len(match.group(1)) == 1:
            continue
        if match:
            if current:
                current["end"] = number
                sections.append(current)
            if len(match.group(1)) == 2:
                major, minor = major + 1, 0
                label = f"{major:02d}"
            else:
                minor += 1
                label = f"{major:02d}-{minor}"
            current = {
                "heading": match.group(2),
                "no": label,
                "start": number + 1,
                "end": None,
            }
        elif (
            current is None
            and lines[number].strip()
            and not lines[number].startswith(">")
        ):
            current = {
                "heading": "Introduction",
                "no": "00",
                "start": number,
                "end": None,
            }
    if current:
        current["end"] = len(lines)
        sections.append(current)
    return {
        "origin": file,
        **{k: v for k, v in meta.items() if k in FRONT_KEYS},
        **(extra or {}),
        "doc_id": str(meta["doc_id"]),
        "pack": pack,
        "scope": scope,
        "id_style": id_style,
        "source": file.relative_to(PROJECT_ROOT).as_posix(),
        "lines": lines,
        "sections": sections,
    }


def iso_date(value: str) -> str:
    """openFDA writes 20250101; documents write 2025-01-01. Return the second form."""
    value = str(value or "")
    if re.fullmatch(r"\d{8}", value):
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def render_label(label: dict, target) -> None:
    """Write one openFDA label as markdown, wrapped so line numbers are useful."""
    width = settings()["wrap_width"]
    drug = label["drug"]
    out = [
        "---",
        f"doc_id: label-{drug}",
        f"title: {drug} label (openFDA)",
        "doc_type: drug_label",
        f"version: {label.get('effective_time', '')}",
        f"effective_date: '{iso_date(label.get('effective_time', ''))}'",
        "synthetic: false",
        "reviewed: true",
        "---",
        f"# {drug} label (openFDA)",
        "",
    ]
    for name, text in label["sections"].items():
        out += [f"## {name}", ""]
        for para in [p for p in str(text).split("\n") if p.strip()]:
            out += [*textwrap.wrap(" ".join(para.split()), width), ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")


def load_labels(pack: str, scope: str) -> list:
    raw_dir = PROJECT_ROOT / load_pack("pharma").raw["source"]["raw_dir"]
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No label files in {raw_dir}. Phase 3 Step 3.5 makes them.")
    docs = []
    for file in files:
        label = json.loads(file.read_text(encoding="utf-8"))
        target = path("rendered_dir") / "pharma" / f"{label['drug']}.md"
        render_label(label, target)
        brand = (label.get("brand_name") or [label["drug"]])[0]
        extra = {"drug": label["drug"], "brand": brand, "origin": file}
        docs.append(parse_markdown(target, pack, scope, "label", extra))
    return docs


def load_records(pack: str, scope: str) -> list:
    file = load_pack("hospital").docs_file
    if not file.exists():
        raise SystemExit(f"{file} not found. Phase 2 builds the patient documents.")
    docs = []
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lines = row["text"].splitlines() or [""]
            docs.append(
                {
                    "doc_id": row["id"],
                    "title": f"Patient record {row['id']}",
                    "doc_type": "patient_record",
                    "version": "",
                    "synthetic": True,
                    "reviewed": True,
                    "pack": pack,
                    "scope": scope,
                    "id_style": "record",
                    "source": file.relative_to(PROJECT_ROOT).as_posix(),
                    "origin": file,
                    "lines": lines,
                    "sections": [
                        {"heading": "Record", "no": "01", "start": 0, "end": len(lines)}
                    ],
                }
            )
    return docs


def load_documents(pack: str) -> list:
    docs = []
    for source in packs()[pack]["sources"]:
        kind, scope = source["type"], source["scope"]
        if kind == "markdown_dir":
            folder = PROJECT_ROOT / source["path"]
            for file in sorted(folder.glob("*.md")):
                docs.append(parse_markdown(file, pack, scope))
        elif kind == "pharma_labels":
            docs += load_labels(pack, scope)
        elif kind == "patient_records":
            docs += load_records(pack, scope)
        else:
            raise SystemExit(f"Unknown source type '{kind}' in rag.yaml")
    seen = set()
    for doc in docs:
        if doc["doc_id"] in seen:
            raise SystemExit(f"Pack {pack}: doc_id '{doc['doc_id']}' is used twice")
        seen.add(doc["doc_id"])
    return docs


def _cut(lines: list, limit: int) -> int:
    """Where to end a chunk: last blank line, else last sentence end, else all."""
    size, best_blank, best_sentence = 0, None, None
    for i, (_, text) in enumerate(lines):
        size += len(text) + 1
        if size > limit and i > 0:
            break
        if not text.strip() and i > 0:
            best_blank = i
        elif SENTENCE_END.search(text):
            best_sentence = i + 1
    else:
        return len(lines)
    return best_blank or best_sentence or max(i, 1)


def chunk_document(doc: dict) -> list:
    style = doc["id_style"]
    limit = settings()["label_chunk_chars" if style == "label" else "chunk_chars"]
    overlap = settings()["overlap_lines"]
    chunks = []
    for section in doc["sections"]:
        body = [(n, doc["lines"][n]) for n in range(section["start"], section["end"])]
        while body and not body[0][1].strip():
            body.pop(0)
        while body and not body[-1][1].strip():
            body.pop()
        number = 0
        while body:
            cut = len(body) if style == "record" else _cut(body, limit)
            part = body[:cut]
            text = "\n".join(t for _, t in part).strip()
            if text:
                number += 1
                chunks.append(_make_chunk(doc, section, number, part, text))
            rest = body[cut:]
            while rest and not rest[0][1].strip():
                rest.pop(0)
            at_paragraph = cut < len(body) and not body[cut][1].strip()
            # Overlap must be shorter than the chunk, or the next chunk would
            # start where this one did and the loop would never end.
            keep = min(overlap, cut - 1)
            if rest and not at_paragraph and style != "record" and keep > 0:
                tail = [x for x in part if x[1].strip()][-keep:]
                rest = tail + rest
            if len(rest) >= len(body):
                where = f"{doc['doc_id']} section {section['heading']}"
                raise RuntimeError(f"{where}: chunker made no progress")
            body = rest
    return chunks


def _make_chunk(doc: dict, section: dict, number: int, part: list, text: str) -> dict:
    style = doc["id_style"]
    if style == "label":
        cid = f"{doc['drug']}.{section['heading']}.{number:02d}"
        header = (
            f"Drug: {doc['drug']} | Brand: {doc.get('brand', '')} | "
            f"Section: {section['heading'].replace('_', ' ')} | "
            f"Label date: {doc.get('version', '')}"
        )
    elif style == "record":
        cid, header = doc["doc_id"], ""
    else:
        cid = f"{doc['doc_id']}.s{section['no']}.{number:02d}"
        header = f"{doc['title']} | {section['heading']}"
    first, last = part[0][0] + 1, part[-1][0] + 1
    return {
        "id": cid,
        "text": f"{header}\n{text}" if header else text,
        "pack": doc["pack"],
        "scope": doc["scope"],
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "doc_type": doc.get("doc_type", ""),
        "version": str(doc.get("version", "")),
        "effective_date": str(doc.get("effective_date", "")),
        "section": section["heading"],
        "section_no": section["no"],
        "line_start": first,
        "line_end": last,
        "source": doc["source"],
        "origin": doc["origin"].relative_to(PROJECT_ROOT).as_posix(),
        "synthetic": bool(doc.get("synthetic", True)),
        "drug": doc.get("drug", ""),
    }


def citation_label(chunk: dict) -> str:
    """Human-readable citation: pack, document, section and lines."""
    lines = f"lines {chunk['line_start']}-{chunk['line_end']}"
    if chunk["doc_type"] == "patient_record":
        where = f"patient record {chunk['doc_id']}, {lines}"
    else:
        where = (
            f'{chunk["doc_id"]} "{chunk["title"]}", section '
            f"{chunk['section_no'].lstrip('0') or '0'} {chunk['section']}, {lines}"
        )
    return f"{chunk['pack']} | {where}"


def main():
    utf8_stdout()
    if len(sys.argv) != 2 or sys.argv[1] not in packs():
        raise SystemExit(f"Usage: python -m core.rag.documents <{'|'.join(packs())}>")
    pack = sys.argv[1]
    docs = load_documents(pack)
    chunks = [c for d in docs for c in chunk_document(d)]
    if not chunks:
        raise SystemExit(f"Pack {pack}: no chunks. Are its documents generated?")
    sizes = sorted(len(c["text"]) for c in chunks)
    unreviewed = [d["doc_id"] for d in docs if not d.get("reviewed", True)]
    print(f"Pack       : {pack}")
    print(f"Documents  : {len(docs)}   unreviewed: {len(unreviewed)}")
    print(
        f"Chunks     : {len(chunks)}   chars min {sizes[0]}, "
        f"median {sizes[len(sizes) // 2]}, max {sizes[-1]}"
    )
    for scope in sorted({c["scope"] for c in chunks}):
        print(f"  scope {scope:10} {sum(c['scope'] == scope for c in chunks)} chunks")
    sample = next((c for c in chunks if c["doc_type"] != "patient_record"), chunks[0])
    print(f"\nSample chunk: {sample['id']}\nCitation    : {citation_label(sample)}")
    print("-" * 60)
    print(sample["text"][:500])


if __name__ == "__main__":
    main()
