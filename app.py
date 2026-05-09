"""Minimal RAG pipeline (PDF-only) with a Flask chat GUI and follow-up history.

Reads the persistent vector store built by vectorize.py, wires it up with
BM25 + vector hybrid retrieval and a small Qwen Instruct model. Follow-up
questions are resolved against the ongoing chat history: a rewriting step
turns pronouns like "seine Frau" into a standalone query before retrieval.

Run:
    python vectorize.py    # once, to build ./store
    python app.py          # start the web GUI
"""

from __future__ import annotations

import os
import pickle
import threading

import torch
from flask import Flask, jsonify, render_template_string, request
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from vectorize import CHROMA_DIR, CHUNKS_PATH, EMBED_MODEL

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, which may reference earlier turns, "
     "rewrite it as a standalone question understandable on its own. "
     "Do NOT answer. If the question already stands alone, return it verbatim."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer strictly from the provided context. "
     "Reply in one short sentence. Quote numbers exactly as they appear. "
     "If the answer is not in the context, reply exactly: I do not know.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def load_retriever():
    if not os.path.isdir(CHROMA_DIR) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Vector store not found. Run `python vectorize.py` first."
        )
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"[rag] loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    ).as_retriever(search_type="similarity", search_kwargs={"k": 6})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6

    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.6, 0.4])


def build_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.generation_config.max_length = 2048
    print(f"[llm] loaded {LLM_MODEL} on {model.device}")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False,
        eos_token_id=[tokenizer.eos_token_id, im_end_id],
        pad_token_id=tokenizer.eos_token_id,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen), tokenizer=tokenizer)


def build_chain():
    retriever = load_retriever()
    llm = build_llm()
    history_aware = create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_PROMPT)
    combine = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(history_aware, combine)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG Chat</title>
  <style>
    :root { --fg:#1f2328; --muted:#6b7280; --bg:#f6f8fa; --card:#fff; --accent:#2563eb; --border:#d0d7de;
            --user:#dbeafe; --bot:#f0f3f6; --err:#b91c1c; }
    * { box-sizing: border-box; }
    body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }
    main { max-width: 820px; margin: 2rem auto; padding: 0 1rem; }
    header { display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; }
    h1 { margin:0; font-size:1.4rem; }
    .muted { color:var(--muted); font-size:.9rem; }
    #transcript { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; min-height:100px; max-height:60vh; overflow-y:auto; }
    #transcript:empty::before { content:"No messages yet. Ask a question below."; color:var(--muted); }
    .turn { margin:.35rem 0; display:flex; }
    .turn.user { justify-content:flex-end; }
    .bubble { max-width:82%; padding:.55rem .85rem; border-radius:14px; white-space:pre-wrap; word-wrap:break-word; }
    .user .bubble { background:var(--user); border-bottom-right-radius:4px; }
    .bot  .bubble { background:var(--bot);  border-bottom-left-radius:4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .bot .sources { margin-top:.3rem; font-size:.8rem; color:var(--muted); word-break:break-all; }
    .err .bubble { background:#fee2e2; color:var(--err); }
    form { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:.75rem 1rem; margin-top:1rem; }
    textarea { width:100%; min-height:70px; padding:.5rem .6rem; font:inherit; border:1px solid var(--border); border-radius:8px; resize:vertical; }
    textarea:focus { outline:2px solid var(--accent); outline-offset:-1px; border-color:transparent; }
    .row { display:flex; gap:.5rem; align-items:center; margin-top:.55rem; }
    button { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:.5rem .95rem; font:inherit; cursor:pointer; }
    button.secondary { background:transparent; color:var(--muted); border:1px solid var(--border); }
    button:disabled { opacity:.6; cursor:progress; }
    .spacer { flex:1; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>RAG Chat</h1>
    <span class="muted">{{ model }}</span>
  </header>

  <div id="transcript"></div>

  <form id="f">
    <label for="q" class="muted">Follow-up (Ctrl/Cmd + Enter to send)</label>
    <textarea id="q" placeholder="Ask a follow-up…" autofocus></textarea>
    <div class="row">
      <button id="btn" type="submit">Send</button>
      <button id="clear" type="button" class="secondary">Clear chat</button>
      <span class="spacer"></span>
      <span id="status" class="muted"></span>
    </div>
  </form>
</main>
<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const btn = document.getElementById('btn');
const clearBtn = document.getElementById('clear');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');

function addTurn(role, text, sources) {
  const turn = document.createElement('div');
  turn.className = 'turn ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  turn.appendChild(bubble);
  if (role === 'bot' && sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.textContent = 'sources: ' + sources.join(', ');
    bubble.appendChild(s);
  }
  transcript.appendChild(turn);
  transcript.scrollTop = transcript.scrollHeight;
  return bubble;
}

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text) return;
  addTurn('user', text);
  q.value = '';
  btn.disabled = true; clearBtn.disabled = true; status.textContent = 'Thinking…';
  const placeholder = addTurn('bot', '…');
  try {
    const r = await fetch('/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:text})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
    placeholder.textContent = data.answer || '(empty)';
    if (data.sources && data.sources.length) {
      const s = document.createElement('div');
      s.className = 'sources';
      s.textContent = 'sources: ' + data.sources.join(', ');
      placeholder.appendChild(s);
    }
  } catch (err) {
    placeholder.parentElement.classList.add('err');
    placeholder.textContent = String(err);
  } finally {
    btn.disabled = false; clearBtn.disabled = false; status.textContent = '';
    q.focus();
  }
});

clearBtn.addEventListener('click', async () => {
  clearBtn.disabled = true;
  try { await fetch('/reset', {method:'POST'}); }
  finally {
    transcript.innerHTML = '';
    clearBtn.disabled = false;
    q.focus();
  }
});

q.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') f.requestSubmit();
});
</script>
</body>
</html>
"""


def create_app(chain):
    app = Flask(__name__)
    lock = threading.Lock()
    history: list = []

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, model=LLM_MODEL)

    @app.post("/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify(error="empty question"), 400
        with lock:
            result = chain.invoke({"input": question, "chat_history": list(history)})
            answer = (result["answer"] or "").strip()
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))
        sources = sorted({d.metadata.get("source", "?") for d in result["context"]})
        return jsonify(answer=answer, sources=sources)

    @app.post("/reset")
    def reset():
        with lock:
            history.clear()
        return jsonify(ok=True)

    return app


def main():
    chain = build_chain()
    app = create_app(chain)
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_PORT", "5000"))
    print(f"[rag] GUI ready at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
