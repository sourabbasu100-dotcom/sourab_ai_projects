"""Local PDF RAG backed by ChromaDB and Sentence Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_COLLECTION = "pdf_knowledge"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class Embedder(Protocol):
    def encode(
        self, sentences: Sequence[str], *, normalize_embeddings: bool
    ) -> Any: ...


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int
    distance: float | None


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character chunks, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        word_boundary = text.rfind(" ", 0, next_start)
        start = word_boundary + 1 if word_boundary >= start else next_start
    return chunks


def load_embedder(model_name: str) -> Embedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name)


class PDFRAG:
    def __init__(
        self,
        db_dir: Path,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is not installed. Run: pip install -r requirements.txt"
                ) from exc
            client = chromadb.PersistentClient(path=str(db_dir))

        self.embedder = embedder or load_embedder(DEFAULT_EMBEDDING_MODEL)
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_pdf(
        self,
        pdf_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is not installed. Run: pip install -r requirements.txt"
            ) from exc

        path = pdf_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(str(path))
        source = str(path)
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(
                chunk_text(page_text, chunk_size, overlap)
            ):
                digest = hashlib.sha256(
                    f"{source}\0{page_number}\0{chunk_number}".encode("utf-8")
                ).hexdigest()
                ids.append(digest)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": source,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {path}")

        vectors = self.embedder.encode(documents, normalize_embeddings=True)
        embeddings = vectors.tolist() if hasattr(vectors, "tolist") else vectors

        # Replace this PDF's prior chunks so re-ingestion cannot leave stale text.
        self.collection.delete(where={"source": source})
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(documents)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.collection.count() == 0:
            raise ValueError("The ChromaDB collection is empty; ingest a PDF first")

        vector = self.embedder.encode([query], normalize_embeddings=True)
        query_embedding = vector.tolist() if hasattr(vector, "tolist") else vector
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=document,
                    source=metadata["source"],
                    page=metadata["page"],
                    distance=distance,
                )
            )
        return chunks

    def answer(
        self,
        query: str,
        top_k: int = 4,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.search(query, top_k)
        context = "\n\n".join(
            f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}"
            for chunk in chunks
        )
        payload = {
            "model": llm_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied context. If it does not "
                        "contain the answer, say so. Cite supporting sources with "
                        "their file name and page number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        }
        request = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {ollama_url}: {exc.reason}. "
                "Make sure Ollama is running and the model is available."
            ) from exc

        try:
            answer = result["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an unexpected response") from exc
        return answer, chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB and ask questions with local RAG."
    )
    parser.add_argument("--db-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="ingest one or more PDFs")
    ingest_parser.add_argument("pdfs", nargs="+", type=Path)

    ask_parser = subparsers.add_parser("ask", help="retrieve context and ask Ollama")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ask_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rag = PDFRAG(
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedder=load_embedder(args.embedding_model),
    )

    if args.command == "ingest":
        for pdf_path in args.pdfs:
            count = rag.ingest_pdf(pdf_path)
            print(f"Ingested {count} chunks from {pdf_path}")
    else:
        answer, sources = rag.answer(
            args.question,
            top_k=args.top_k,
            ollama_url=args.ollama_url,
            llm_model=args.llm_model,
        )
        print(answer)
        print("\nRetrieved sources:")
        for chunk in sources:
            print(f"- {chunk.source}, page {chunk.page}")


if __name__ == "__main__":
    main()
