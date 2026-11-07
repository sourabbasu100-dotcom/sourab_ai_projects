import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag import PDFRAG, chunk_text


class FakeEmbedder:
    def encode(self, sentences, *, normalize_embeddings):
        self.last_sentences = sentences
        self.normalized = normalize_embeddings
        return [[float(len(sentence)), 1.0] for sentence in sentences]


class FakeCollection:
    def __init__(self):
        self.records = {}

    def delete(self, *, where):
        self.records = {
            key: value
            for key, value in self.records.items()
            if value["metadata"]["source"] != where["source"]
        }

    def add(self, *, ids, documents, metadatas, embeddings):
        for record_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            self.records[record_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def count(self):
        return len(self.records)


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, **_kwargs):
        return self.collection


class ChunkTextTests(unittest.TestCase):
    def test_normalizes_whitespace_and_keeps_chunks_within_size(self):
        chunks = chunk_text(" one   two\nthree four five ", chunk_size=10, overlap=2)

        self.assertEqual(chunks, ["one two", "two three", "three four", "four five"])
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(" \n\t "), [])

    def test_rejects_invalid_chunk_configuration(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=5, overlap=5)


class IngestionTests(unittest.TestCase):
    def test_ingestion_replaces_existing_pdf_chunks_and_preserves_page_metadata(self):
        collection = FakeCollection()
        embedder = FakeEmbedder()
        rag = PDFRAG(
            db_dir=Path("unused"),
            client=FakeClient(collection),
            embedder=embedder,
        )
        fake_reader = SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: "First page content")]
        )
        fake_pypdf = SimpleNamespace(PdfReader=lambda _path: fake_reader)

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "guide.pdf"
            pdf_path.touch()
            with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
                first_count = rag.ingest_pdf(pdf_path, chunk_size=12, overlap=2)
                first_ids = set(collection.records)
                second_count = rag.ingest_pdf(pdf_path, chunk_size=100, overlap=10)

        self.assertGreater(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(len(collection.records), 1)
        self.assertNotEqual(first_ids, set(collection.records))
        only_record = next(iter(collection.records.values()))
        self.assertEqual(only_record["metadata"]["page"], 1)
        self.assertTrue(embedder.normalized)


if __name__ == "__main__":
    unittest.main()
