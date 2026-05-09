"""Build the persistent vector store for the RAG pipeline.

Loads every PDF next to this file, normalizes whitespace, chunks, filters
near-empty chunks, and writes:

    ./store/chroma/        -- persistent Chroma vector DB
    ./store/chunks.pkl     -- pickled chunks (used by app.py to rebuild BM25)

Run once after placing PDFs, and again whenever PDFs change:

    python vectorize.py
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = HERE
STORE_DIR = os.path.join(HERE, "store")
CHROMA_DIR = os.path.join(STORE_DIR, "chroma")
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.pkl")

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _normalize(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def load_and_chunk_pdfs(pdf_dir: str):
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    pages = []
    for path in pdfs:
        for page in PyPDFLoader(path).load():
            page.page_content = _normalize(page.page_content)
            pages.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return [c for c in chunks if sum(ch.isalnum() for ch in c.page_content) >= 40]


def build():
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(STORE_DIR, exist_ok=True)

    chunks = load_and_chunk_pdfs(PDF_DIR)
    print(f"[vectorize] {len(chunks)} chunks from PDFs in {PDF_DIR}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[vectorize] wrote {CHUNKS_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"[vectorize] wrote Chroma store at {CHROMA_DIR}")


if __name__ == "__main__":
    build()
