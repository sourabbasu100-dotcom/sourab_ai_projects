"""Local HTTP API for the React PDF RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import PDFRAG, RetrievedChunk, load_embedder


ROOT_DIR = Path(__file__).resolve().parent
DB_DIR = Path(os.getenv("RAG_DB_DIR", str(ROOT_DIR / "chroma_db")))
COLLECTION = os.getenv("RAG_COLLECTION", "pdf_knowledge")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

app = FastAPI(title="Local PDF RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_rag: PDFRAG | None = None


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


def get_rag() -> PDFRAG:
    global _rag
    if _rag is None:
        _rag = PDFRAG(
            db_dir=DB_DIR,
            collection_name=COLLECTION,
            embedder=load_embedder(
                os.getenv(
                    "EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            ),
        )
    return _rag


def serialize_sources(chunks: list[RetrievedChunk]) -> list[dict[str, str | int]]:
    return [{"source": chunk.source, "page": chunk.page} for chunk in chunks]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents")
def ingest_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    documents_dir = ROOT_DIR / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    destination = documents_dir / filename
    try:
        with destination.open("wb") as saved_file:
            while content := file.file.read(1024 * 1024):
                saved_file.write(content)
        count = get_rag().ingest_pdf(destination)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"filename": filename, "chunks": count}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    try:
        answer, sources = get_rag().answer(
            request.question,
            top_k=request.top_k,
            ollama_url=OLLAMA_URL,
            llm_model=LLM_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"answer": answer, "sources": serialize_sources(sources)}
