from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import db
from app.ingest import chunk_text, embed_texts
from app.models import IngestResponse, QueryRequest, QueryResponse, SourceChunk
from app.rag import answer_question


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="LLM Knowledge Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="file must be utf-8 text") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="empty file")

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    doc_id = db.upsert_document(file.filename)
    rows = [(idx, chunk, vectors[idx]) for idx, chunk in enumerate(chunks)]
    db.replace_chunks(doc_id, rows)

    return IngestResponse(document=file.filename, chunks_indexed=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    answer, hits = answer_question(body.question, body.top_k)
    sources = [
        SourceChunk(
            document=hit["document"],
            chunk_index=hit["chunk_index"],
            score=round(hit["score"], 4),
            excerpt=hit["content"][:280],
        )
        for hit in hits
    ]
    return QueryResponse(answer=answer, sources=sources)
