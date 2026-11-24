"""FastAPI semantic search service."""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field

from .embeddings import LocalEmbedder, HashEmbedder
from .chunking import build_chunks, stable_chunk_id
from .vector_store import QdrantStore

load_dotenv()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: Optional[str] = None
    min_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


def verify_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY", "week6-demo-key")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    import secrets
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True



@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("WEEK6_TEST_MODE") == "1":
        app.state.embedder = HashEmbedder()
        app.state.store = QdrantStore(in_memory=True)

        demo_chunks = [
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "text": "Employees may request annual leave through the HR portal.",
                "metadata": {
                    "source": "leave_policy.txt",
                    "page": 1,
                    "section": "LEAVE POLICY",
                    "chunk_index": 0,
                    "doc_type": "txt"
                }
            },
            {
                "id": "22222222-2222-4222-8222-222222222222",
                "text": "Password resets are available from the account security page.",
                "metadata": {
                    "source": "it_manual.txt",
                    "page": 2,
                    "section": "ACCOUNT SECURITY",
                    "chunk_index": 0,
                    "doc_type": "txt"
                }
            }
        ]

        vectors = app.state.embedder.encode(
            [chunk["text"] for chunk in demo_chunks]
        )

        app.state.store.upsert(demo_chunks, vectors)

    else:
        app.state.embedder = LocalEmbedder()
        app.state.store = QdrantStore()

        docs_dir = Path("data/documents")

        paths = (
            list(docs_dir.glob("*.txt"))
            + list(docs_dir.glob("*.pdf"))
            + list(docs_dir.glob("*.docx"))
        )

        if paths and app.state.store.count() == 0:
            chunks = build_chunks(paths)

            vectors = app.state.embedder.encode(
                [chunk["text"] for chunk in chunks]
            )

            app.state.store.upsert(chunks, vectors)

    yield

    app.state.store.close()



app = FastAPI(title="Week 6 Semantic Search API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "week6-semantic-search"}


@app.post("/search", dependencies=[Depends(verify_api_key)])
def search(request: SearchRequest):
    vector = app.state.embedder.encode([request.query])[0]
    results = app.state.store.search(
        vector, top_k=request.top_k, doc_type=request.doc_type,
        min_score=request.min_score
    )
    return {"query": request.query, "count": len(results), "results": results}


@app.post("/documents", dependencies=[Depends(verify_api_key)])
async def upload_and_index(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT and MD are supported")
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    try:
        chunks = build_chunks([temp_path])
        # Preserve the uploaded filename in metadata and stable IDs.
        for chunk in chunks:
            chunk["metadata"]["source"] = file.filename
            chunk["id"] = stable_chunk_id(file.filename, chunk["metadata"]["page"],
                                           chunk["metadata"]["section"], chunk["metadata"]["chunk_index"])
        vectors = app.state.embedder.encode([c["text"] for c in chunks])
        app.state.store.upsert(chunks, vectors)
        return {"filename": file.filename, "chunks_indexed": len(chunks),
                "total_vectors": app.state.store.count()}
    finally:
        temp_path.unlink(missing_ok=True)
