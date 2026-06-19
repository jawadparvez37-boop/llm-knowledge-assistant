from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if overlap >= size:
        overlap = max(0, size // 5)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
