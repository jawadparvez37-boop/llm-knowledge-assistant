# LLM Knowledge Assistant

Document Q&A API backed by PostgreSQL and pgvector.

## Stack

- FastAPI
- OpenAI embeddings
- PostgreSQL / pgvector

## Quick start

```bash
cp .env.example .env
docker compose up -d
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Ingest sample docs:

```bash
python scripts/ingest_sample.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload a text file |
| POST | `/query` | Ask a question against indexed docs |

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Layout

```
app/
  main.py      API routes
  rag.py       retrieval + generation
  ingest.py    chunking and indexing
  db.py        pgvector storage
  config.py    settings
  models.py    request/response schemas
```
