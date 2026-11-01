"""Phase 3: link drug-label names to Synthea medications, patients and person_id."""

import re
import sys

import pandas as pd

from src.common.config import PROJECT_ROOT, get_settings
from src.phase2.check_claims import load_records
from src.phase3.packs import load_pack
from src.phase3.scoring import patient_has

OUT_DIR = PROJECT_ROOT / "data" / "dictionary"
WORLD = PROJECT_ROOT / "data" / "world" / "person.parquet"


def build():
    pack = load_pack("pharma")
    drugs = pack.raw["source"]["drugs"]
    s = get_settings()
    meds = pd.read_csv(
        s.synthea_dir / "medications.csv",
        usecols=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
    )
    lowered = meds["DESCRIPTION"].str.lower()

    parts = []
    for drug in drugs:
        mask = lowered.str.contains(rf"\b{re.escape(drug)}", regex=True)
        part = meds[mask].copy()
        part["drug"] = drug
        part["combo"] = part["DESCRIPTION"].str.contains(" / ", regex=False)
        parts.append(part)
    found = pd.concat(parts, ignore_index=True)
    found["active"] = found["STOP"].isna()

    person = pd.read_parquet(WORLD, columns=["person_id", "synthea_id"])
    found = found.merge(person, left_on="PATIENT", right_on="synthea_id", how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drug_map = (
        found.groupby(["drug", "DESCRIPTION", "CODE", "combo"])
        .agg(rows=("PATIENT", "size"), patients=("PATIENT", "nunique"))
        .reset_index()
    )
    drug_map.to_csv(OUT_DIR / "drug_map.csv", index=False)
    found = found.drop(columns=["synthea_id"])
    found.to_csv(OUT_DIR / "patient_drugs.csv", index=False)

    print(f"Medication rows read : {len(meds)}")
    print(f"Rows matching a drug : {len(found)}")
    print(f"Rows with a person_id: {int(found['person_id'].notna().sum())}\n")
    summary = (
        found.groupby("drug")
        .agg(
            patients=("PATIENT", "nunique"),
            active_rows=("active", "sum"),
            combo_rows=("combo", "sum"),
        )
        .reindex(drugs)
        .fillna(0)
        .astype(int)
    )
    print(summary.to_string())
    print(f"\nWrote {OUT_DIR / 'drug_map.csv'}")
    print(f"Wrote {OUT_DIR / 'patient_drugs.csv'}")


def query(drug: str, condition: str | None):
    path = OUT_DIR / "patient_drugs.csv"
    if not path.exists():
        raise SystemExit("Run: python -m src.phase3.drug_dictionary build")
    table = pd.read_csv(path)
    sub = table[table["drug"] == drug.lower()]
    patients = sub.drop_duplicates("PATIENT")
    active = sub[sub["active"]].drop_duplicates("PATIENT")
    print(f"Drug              : {drug.lower()}")
    print(f"Patients          : {len(patients)}")
    print(f"With an active Rx : {len(active)}")
    print("\nFirst 10 patients (Synthea ID, person_id, example product):")
    for row in patients.head(10).itertuples():
        print(f"  {row.PATIENT}  {row.person_id}  {row.DESCRIPTION}")
    if condition:
        records = load_records()
        both = [
            pid
            for pid in patients["PATIENT"]
            if pid in records and patient_has(records[pid], condition)
        ]
        print(f"\nOf these, with '{condition}' in the hospital records: {len(both)}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(
            "Usage: python -m src.phase3.drug_dictionary build\n"
            "       python -m src.phase3.drug_dictionary query <drug> "
            '[--condition "<text>"]'
        )
    if args[0] == "build":
        build()
    elif args[0] == "query" and len(args) >= 2:
        condition = None
        if "--condition" in args:
            condition = args[args.index("--condition") + 1]
        query(args[1], condition)
    else:
        raise SystemExit("Unknown command. Use build or query.")


if __name__ == "__main__":
    main()
