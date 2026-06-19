from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=12)


class SourceChunk(BaseModel):
    document: str
    chunk_index: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    document: str
    chunks_indexed: int
