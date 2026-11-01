"""Central config loader: every phase reads settings from here, never hard-codes them."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    chat_model: str
    embed_model: str
    synthea_dir: Path | None


def get_settings() -> Settings:
    raw_dir = os.getenv("SYNTHEA_DIR")
    return Settings(
        chat_model=os.getenv("CHAT_MODEL", ""),
        embed_model=os.getenv("EMBED_MODEL", ""),
        synthea_dir=Path(raw_dir) if raw_dir else None,
    )


if __name__ == "__main__":
    s = get_settings()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Chat model   : {s.chat_model}")
    print(f"Embed model  : {s.embed_model}")
    print(
        f"Synthea dir  : {s.synthea_dir}  exists={s.synthea_dir.exists() if s.synthea_dir else False}"
    )
