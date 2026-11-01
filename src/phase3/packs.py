"""Phase 3: pack configuration. A pack is one YAML file in core/packs."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.common.config import PROJECT_ROOT

PACKS_DIR = PROJECT_ROOT / "core" / "packs"
DB_DIR = PROJECT_ROOT / "data" / "vectordb"
REQUIRED = (
    "name",
    "title",
    "collection",
    "docs_file",
    "id_pattern",
    "system_prompt",
    "test_file",
)


@dataclass(frozen=True)
class Pack:
    name: str
    title: str
    collection: str
    docs_file: Path
    top_k: int
    id_pattern: re.Pattern
    system_prompt: str
    test_file: Path
    raw: dict = field(default_factory=dict, compare=False, hash=False)


def list_packs() -> list:
    return sorted(p.stem for p in PACKS_DIR.glob("*.yaml"))


def load_pack(name: str) -> Pack:
    path = PACKS_DIR / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"No pack file: {path}. Available: {list_packs()}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise SystemExit(f"Pack '{name}' is missing keys: {missing}")
    return Pack(
        name=cfg["name"],
        title=cfg["title"],
        collection=cfg["collection"],
        docs_file=PROJECT_ROOT / cfg["docs_file"],
        top_k=int(cfg.get("top_k", 5)),
        id_pattern=re.compile(cfg["id_pattern"]),
        system_prompt=cfg["system_prompt"].strip(),
        test_file=PROJECT_ROOT / cfg["test_file"],
        raw=cfg,
    )


def collection_count(pack: Pack):
    """Number of items in the pack's Chroma collection, or None if it is missing."""
    import chromadb
    from chromadb.errors import NotFoundError

    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        return client.get_collection(pack.collection).count()
    except NotFoundError:
        return None


def main():
    print(f"{'pack':10} {'collection':12} {'docs file':6} {'indexed':8} title")
    for name in list_packs():
        pack = load_pack(name)
        docs = "yes" if pack.docs_file.exists() else "no"
        count = collection_count(pack)
        indexed = "-" if count is None else str(count)
        print(f"{pack.name:10} {pack.collection:12} {docs:6} {indexed:8} {pack.title}")


if __name__ == "__main__":
    main()
