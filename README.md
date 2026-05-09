# rag_app

A minimal local-only RAG pipeline over the PDFs sitting next to the code. Hybrid (BM25 + vector) retrieval, follow-up-aware question rewriting, and a small Qwen Instruct model for the answer — all wrapped in a tiny Flask chat UI.

## Layout

```
vectorize.py     # one-shot indexer: PDFs -> chunks -> Chroma + pickled chunks
app.py           # Flask GUI + LangChain retrieval chain
store/           # persisted index (created by vectorize.py, gitignored)
*.pdf            # source documents (drop your own here)
```

## How it works

`vectorize.py`:

- Loads every `*.pdf` next to itself with `PyPDFLoader`.
- Normalises whitespace, splits into 1000-char chunks with 150-char overlap.
- Drops near-empty chunks (`<40` alphanumeric chars).
- Embeds with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Persists a Chroma DB to `store/chroma/` and pickles the chunk list to `store/chunks.pkl`.

`app.py`:

- Reloads the Chroma store and rebuilds a BM25 retriever from the pickled chunks.
- Combines them in an `EnsembleRetriever` (BM25 0.6 / vector 0.4, top-k=6 each).
- Loads `Qwen/Qwen2.5-1.5B-Instruct` (bf16 on CUDA, fp32 on CPU).
- Wires a history-aware retriever: a contextualisation prompt rewrites follow-up questions like "and his wife?" into standalone queries before retrieval.
- Final QA prompt is strict: answer in one short sentence, quote numbers verbatim, reply `I do not know.` if the answer isn't in context.
- Serves a single-page chat UI at `/` with `/ask` and `/reset` endpoints.

## Run

```bash
pip install flask torch transformers \
            langchain-chroma langchain-classic langchain-community \
            langchain-core langchain-huggingface langchain-text-splitters

python vectorize.py    # build store/ — re-run whenever PDFs change
python app.py          # serve the GUI
```

Open http://127.0.0.1:5000.

Override host/port with `RAG_HOST` / `RAG_PORT`.

## Notes

- Multilingual embeddings — works for German PDFs (the included `Einkommensteuerbescheid.pdf` is gitignored as a personal example).
- `store/` is rebuilt from scratch on every `vectorize.py` run; safe to delete.
- Single-process Flask with an in-memory `history` list guarded by a `Lock` — fine for local use, not for multi-user deployment.
