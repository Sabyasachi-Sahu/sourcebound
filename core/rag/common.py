"""Phase 5: shared settings, model calls and text helpers for core/rag."""

import hashlib
import json
import os
import re
import sys
from functools import lru_cache

import ollama
import yaml

from src.common.config import PROJECT_ROOT, get_settings

CONFIG_FILE = PROJECT_ROOT / "core" / "rag" / "rag.yaml"
WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
HYPHENS = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
STOPWORDS = """
a an and are as at be been by can could did do does for from had has have how i if
in into is it its may might must not of on or our shall should so than that the
their them then there these they this those to was we were what when where which
who whom why will with would you your about after before each any all per
"""
STOP = set(STOPWORDS.split())


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def settings() -> dict:
    return config()["settings"]


def packs() -> dict:
    return config()["packs"]


def path(key: str):
    return PROJECT_ROOT / settings()[key]


def config_hash() -> str:
    return hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest()[:12]


def chat_model() -> str:
    return (
        getattr(get_settings(), "chat_model", None)
        or os.getenv("CHAT_MODEL")
        or "gpt-oss:20b-cloud"
    )


def embed_model() -> str:
    return (
        getattr(get_settings(), "embed_model", None)
        or os.getenv("EMBED_MODEL")
        or "nomic-embed-text"
    )


def chat(messages: list, temperature: float = 0) -> str:
    response = ollama.chat(
        model=chat_model(), messages=messages, options={"temperature": temperature}
    )
    return response["message"]["content"] or ""


def embed(texts: list, kind: str) -> list:
    """Embed texts. kind is 'query' or 'doc' (nomic-embed-text task prefixes)."""
    prefix = settings()[f"embed_{kind}_prefix"]
    out = []
    size = settings()["embed_batch"]
    for i in range(0, len(texts), size):
        batch = [prefix + t for t in texts[i : i + size]]
        out.extend(ollama.embed(model=embed_model(), input=batch)["embeddings"])
    return out


def normalise(text: str) -> str:
    """Unicode hyphens to '-', thousands separators removed from numbers."""
    text = text.translate(HYPHENS)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def tokens(text: str) -> list:
    return [w for w in WORD.findall(normalise(text).lower()) if w not in STOP]


def numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", normalise(text)))


def utf8_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(file) -> list:
    if not file.exists():
        return []
    with file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(file, rows: list):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
