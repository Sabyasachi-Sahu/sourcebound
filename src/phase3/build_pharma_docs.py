"""Phase 3: turn downloaded drug labels into section-aware chunks."""

import json

from src.common.config import PROJECT_ROOT
from src.phase3.chunking import chunk_text
from src.phase3.packs import load_pack


def main():
    pack = load_pack("pharma")
    src = pack.raw["source"]
    raw_dir = PROJECT_ROOT / src["raw_dir"]
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No label files in {raw_dir}. Run scripts\\fetch_pharma_labels.py first."
        )

    records = []
    per_drug = {}
    truncated_sections = 0
    for path in files:
        label = json.loads(path.read_text(encoding="utf-8"))
        drug = label["drug"]
        brand = (label.get("brand_name") or [drug])[0]
        date = label.get("effective_time", "")
        for section, text in label["sections"].items():
            chunks, truncated = chunk_text(
                text,
                size=src["chunk_size"],
                overlap=src["chunk_overlap"],
                max_chars=src["max_chars_per_section"],
            )
            truncated_sections += int(truncated)
            for number, chunk in enumerate(chunks, start=1):
                header = (
                    f"Drug: {drug} | Brand: {brand} | "
                    f"Section: {section.replace('_', ' ')} | Label date: {date}"
                )
                records.append(
                    {
                        "id": f"{drug}.{section}.{number:02d}",
                        "text": f"{header}\n{chunk}",
                        "metadata": {
                            "drug": drug,
                            "section": section,
                            "set_id": label.get("set_id", ""),
                            "effective_time": date,
                            "brand": brand,
                            "route": ", ".join(label.get("route") or []),
                            "chunk_no": number,
                            "truncated": truncated,
                        },
                    }
                )
                per_drug[drug] = per_drug.get(drug, 0) + 1

    out = pack.docs_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(r["text"]) for r in records]
    print(f"Label files        : {len(files)}")
    print(f"Chunks written     : {len(records)}")
    print(f"Chunk length (chars): min {min(lengths)}, max {max(lengths)}")
    print(f"Sections truncated : {truncated_sections}")
    print(f"Output file        : {out}")
    print("\nChunks per drug:")
    for drug, count in sorted(per_drug.items()):
        print(f"  {drug:22} {count}")
    print("\n--- Sample chunk ---")
    print(records[0]["id"])
    print(records[0]["text"][:400])


if __name__ == "__main__":
    main()
