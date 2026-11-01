"""Phase 3: split long label text into overlapping chunks."""

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hard_split(piece: str, size: int) -> list:
    return [piece[i : i + size] for i in range(0, len(piece), size)]


def chunk_text(text: str, size: int = 1200, overlap: int = 150, max_chars: int = 8000):
    """Return (chunks, truncated). Chunks break on sentence ends, with overlap."""
    text = clean(text)
    truncated = len(text) > max_chars
    text = text[:max_chars]

    pieces = []
    for sentence in SENTENCE_END.split(text):
        if len(sentence) > size:
            pieces.extend(hard_split(sentence, size))
        elif sentence:
            pieces.append(sentence)

    chunks, current, length = [], [], 0
    for piece in pieces:
        if current and length + len(piece) + 1 > size:
            chunks.append(" ".join(current))
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            if tail_len + len(piece) + 1 > size:
                tail, tail_len = [], 0
            current, length = tail, tail_len
        current.append(piece)
        length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks, truncated


def main():
    sample = " ".join(
        f"Sentence number {i} says something about a drug." for i in range(120)
    )
    chunks, truncated = chunk_text(sample, size=500, overlap=80, max_chars=4000)
    sizes = [len(c) for c in chunks]
    print(f"input chars : {len(sample)}")
    print(f"chunks      : {len(chunks)}")
    print(f"sizes       : min {min(sizes)}, max {max(sizes)}")
    print(f"truncated   : {truncated}")
    first = set(SENTENCE_END.split(chunks[0]))
    second = set(SENTENCE_END.split(chunks[1]))
    print(f"overlap kept: {'yes' if first & second else 'no'}")


if __name__ == "__main__":
    main()
