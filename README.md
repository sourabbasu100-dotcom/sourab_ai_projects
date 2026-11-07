# PDF RAG with ChromaDB

This project extracts text from PDFs, splits it into overlapping chunks, embeds
the chunks with Sentence Transformers, and stores them in a persistent local
ChromaDB collection. Questions retrieve relevant chunks and send them to a local
Ollama model to generate a context-grounded answer.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default embedding model is downloaded the first time it is used. Install
[Ollama](https://ollama.com/) separately, then download the default chat model:

```powershell
ollama pull qwen2.5:3b
```

## Ingest PDFs

```powershell
python rag.py ingest .\docs\guide.pdf
python rag.py ingest .\docs\another.pdf .\docs\reference.pdf
```

ChromaDB data is saved in `.\chroma_db`. Re-ingesting a PDF replaces its stored
chunks. Scanned/image-only PDFs need OCR before ingestion; this example extracts
embedded PDF text and does not perform OCR.

## Ask questions

```powershell
python rag.py ask "What does the guide say about installation?"
python rag.py ask "Summarize the deployment steps" --top-k 6
```

The CLI prints the generated answer and the retrieved PDF pages. Ollama must be
running locally. Override the database, collection, embedding model, or Ollama
settings with:

```powershell
python rag.py --db-dir .\my_chroma --collection manuals ingest .\docs\guide.pdf
python rag.py --db-dir .\my_chroma --collection manuals ask "How do I install it?" `
  --ollama-url http://localhost:11434 --llm-model qwen2.5:3b
```

## React chatbot

The React interface uploads PDFs and lets you chat with the indexed content. In
one PowerShell window, start the local API:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, start the frontend:

```powershell
cd .\frontend
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`), add a PDF,
and ask questions. Keep Ollama running with the model downloaded above. Uploaded
PDFs are saved in `.\documents`, and the frontend uses the same `.\chroma_db`
collection as the CLI when both are started from the project directory.

## Tests

Run the standard-library unit tests with:

```powershell
python -m unittest -v
```
