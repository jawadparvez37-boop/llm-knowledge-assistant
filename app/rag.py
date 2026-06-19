from openai import OpenAI

from app.config import settings
from app.db import search_similar
from app.ingest import embed_query

_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM = (
    "Answer using only the provided context. "
    "If the context does not contain the answer, say you do not have enough information."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['document']}#{hit['chunk_index']}]\n{hit['content']}"
        )
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No indexed documents matched this question."

    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def answer_question(question: str, top_k: int) -> tuple[str, list[dict]]:
    vector = embed_query(question)
    hits = search_similar(vector, top_k)
    answer = generate_answer(question, hits)
    return answer, hits
