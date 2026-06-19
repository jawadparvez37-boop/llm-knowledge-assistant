from pathlib import Path

from app import db
from app.ingest import chunk_text, embed_texts

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


def ingest_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    doc_id = db.upsert_document(path.name)
    rows = [(idx, chunk, vectors[idx]) for idx, chunk in enumerate(chunks)]
    db.replace_chunks(doc_id, rows)
    return len(chunks)


def main() -> None:
    db.init_db()
    total = 0
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        count = ingest_file(path)
        print(f"{path.name}: {count} chunks")
        total += count
    print(f"done, {total} chunks indexed")


if __name__ == "__main__":
    main()
