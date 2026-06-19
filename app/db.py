import psycopg
from pgvector.psycopg import register_vector

from app.config import settings

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def upsert_document(name: str) -> int:
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (name)
            VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (name,),
        ).fetchone()
        conn.commit()
        return row[0]


def replace_chunks(document_id: int, rows: list[tuple[int, str, list[float]]]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            [(document_id, idx, text, emb) for idx, text, emb in rows],
        )
        conn.commit()


def search_similar(embedding: list[float], top_k: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.name, c.chunk_index, c.content,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, top_k),
        ).fetchall()

    return [
        {
            "document": row[0],
            "chunk_index": row[1],
            "content": row[2],
            "score": float(row[3]),
        }
        for row in rows
    ]
