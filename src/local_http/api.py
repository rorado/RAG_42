from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.bm25 import BM25Search
from rag.chunk_loader import load_chunks
from rag.llm import LLM


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1)


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25 = BM25Search()
    bm25.index(load_chunks())
    app.state.bm25 = bm25
    app.state.llm = LLM()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/search")
def search(request: SearchRequest):
    results = app.state.bm25.search(
        request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "results": [
            {
                "path": chunk.path,
                "start": chunk.start,
                "end": chunk.end,
                "content": chunk.content,
            }
            for chunk in results
        ],
    }


@app.post("/answer")
def answer(request: AnswerRequest):
    results = app.state.bm25.search(request.query, top_k=request.top_k)

    if not results:
        return {
            "message": "No sources found"
        }

    context = "\n\n".join(
        chunk.content for chunk in results
    )

    llm = app.state.llm

    response = llm.answer(request.query, context)

    return {
        "Answer": response
    }

