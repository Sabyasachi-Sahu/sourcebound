"""Download one openFDA label per drug in core/packs/pharma.yaml."""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CONFIG_FILE = ROOT / "core" / "packs" / "pharma.yaml"
CONFIG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
SOURCE = CONFIG["source"]
DRUGS = SOURCE["drugs"]
SECTIONS = SOURCE["sections"]
OUT = ROOT / SOURCE["raw_dir"]
URL = "https://api.fda.gov/drug/label.json"
COMBINATION_MARKS = (" and ", ",", "/", "+", " with ")


def get_results(params: dict, tries: int = 3):
    """Return a list of labels, [] if none match, or None if the request failed."""
    key = os.getenv("OPENFDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(URL, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            print(f"    network error ({type(exc).__name__}), try {attempt}")
            time.sleep(2 * attempt)
            continue
        if r.status_code == 200:
            return r.json().get("results", [])
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            print("    rate limited, waiting")
            time.sleep(5 * attempt)
            continue
        print(f"    HTTP {r.status_code}")
        time.sleep(2)
    return None


def single_ingredient(label: dict, drug: str) -> bool:
    names = [n.lower() for n in label.get("openfda", {}).get("generic_name", [])]
    if not names:
        return False
    return all(drug in n and not any(m in n for m in COMBINATION_MARKS) for n in names)


def section_text(label: dict) -> dict:
    found = {}
    for name in SECTIONS:
        value = label.get(name)
        if value:
            text = "\n".join(value) if isinstance(value, list) else str(value)
            if text.strip():
                found[name] = text
    return found


def pick(results: list, drug: str):
    best, best_key = None, None
    for label in results:
        if not single_ingredient(label, drug):
            continue
        sections = section_text(label)
        key = (len(sections), label.get("effective_time", ""))
        if best_key is None or key > best_key:
            best, best_key = (label, sections), key
    return best


def main():
    force = "--force" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for drug in DRUGS:
        target = OUT / f"{drug}.json"
        if target.exists() and not force:
            print(f"{drug:22} already downloaded (use --force to refresh)")
            ok += 1
            continue
        results = get_results({"search": f'openfda.generic_name:"{drug}"', "limit": 25})
        if results is None:
            print(f"{drug:22} FAILED (request error)")
            failed.append(drug)
            continue
        chosen = pick(results, drug)
        if chosen is None:
            print(f"{drug:22} FAILED (no single-ingredient label found)")
            failed.append(drug)
            continue
        label, sections = chosen
        openfda = label.get("openfda", {})
        record = {
            "drug": drug,
            "set_id": (openfda.get("spl_set_id") or [label.get("set_id", "")])[0],
            "effective_time": label.get("effective_time", ""),
            "version": label.get("version", ""),
            "brand_name": openfda.get("brand_name", []),
            "generic_name": openfda.get("generic_name", []),
            "route": openfda.get("route", []),
            "manufacturer_name": openfda.get("manufacturer_name", []),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sections": sections,
        }
        text = json.dumps(record, indent=2, ensure_ascii=False)
        target.write_text(text, encoding="utf-8")
        brand = (record["brand_name"] or ["?"])[0]
        route = ", ".join(record["route"]) or "?"
        print(
            f"{drug:22} OK  {brand[:24]:24} route={route:10} "
            f"sections={len(sections):2} date={record['effective_time']}"
        )
        ok += 1
        time.sleep(0.5)

    print(f"\nDownloaded or present: {ok} of {len(DRUGS)}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    print(f"Files are in {OUT}")
    print("openFDA data is unvalidated. Do not use it for medical decisions.")


if __name__ == "__main__":
    main()
