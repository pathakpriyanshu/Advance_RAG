# 🧠 PDF‑Q\&A with Hybrid Search & Llama‑3 (70B)

A single‑file Streamlit app that lets you **upload any PDF and ask natural‑language questions** about its content.
Behind the scenes it uses a **hybrid retrieval pipeline** (dense vectors + BM25) and answers with the *Llama‑3 70B* model running on **Groq**.

---



---

## 🏗 Architecture

```text
                          ┌──────────────┐
         Upload PDF ────▶│ PyPDFLoader  │
                          └─────┬────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │ Recursive Splitter │
                     │ (chunk=1000, ov=100) │
                     └─────────┬──────────┘
                               │
        ┌────────────────────┐ ▼ ┌────────────────────────┐
        │ Lowercased Text    │   │ Original Chunked Docs  │
        └────────────────────┘   └────────────────────────┘
                 │                        │
                 ▼                        ▼
      ┌─────────────────────┐   ┌──────────────────────────┐
      │  BM25 Retriever      │   │     Chroma VectorStore   │
      │ (Sparse, k=2)        │   │  (Dense, MiniLM-L6-v2)   │
      └────────┬────────────┘   └────────────┬─────────────┘
               │           Weight 0.3        │ Weight 0.7
               └────────────┬────────────────┘
                            ▼
               ┌────────────────────────────┐
               │   EnsembleRetriever         │
               │  (Hybrid: BM25 + Embedding) │
               └────────────┬───────────────┘
                            ▼
             ┌──────────────────────────────┐
             │ LangChain Retrieval Chain    │
             └────────────┬─────────────────┘
                          ▼
              ┌────────────────────────────┐
              │  LLM: Groq LLaMA3-70B       │
              └────────────┬───────────────┘
                           ▼
             ┌─────────────────────────────┐
             │ Answer + 2 Smart Suggestions│
             └────────────┬────────────────┘
                          ▼
                  📄 Streamlit Interface
```

### 🔍 Why **hybrid**?

| Retriever          | Strength                              | Weakness                     |
| ------------------ | ------------------------------------- | ---------------------------- |
| **Dense (Chroma)** | Semantic matching, typos OK           | Can miss exact keywords      |
| **Sparse (BM25)**  | Exact token hits, great for names/IDs | Case‑sensitive, no semantics |

By weighting them **0.7 (dense) + 0.3 (sparse)** we get the best of both worlds.

### 🦏 Smart lower‑casing trick

BM25 is token‑based **and case‑sensitive**.
We lowercase **both** the corpus and every incoming query via a tiny subclass:

```python
class LowercaseBM25Retriever(BM25Retriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None, **kw):
        return super()._get_relevant_documents(query.lower(), run_manager=run_manager, **kw)
```

Result: *"WHAT IS CHATGPT"* → same hits as *"chatgpt"*.

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone repo & cd
# 2. Put your Groq key in .env
echo "GROQ_API_KEY=sk_..." > .env

# 3. Install deps
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

---




---

## 📂 Project Layout

```text
├── app.py             ← Streamlit front‑end + RAG pipeline
├── requirements.txt   ← pinned libs
├── .env               ← your Groq API key (never commit!)
└── README.md          ← you are here
```

---

## 📜 Prompt Template

```text
Answer the following question only with information found in the context.
Search the entire context before answering.
Give a long, detailed answer.

After answering, suggest two follow‑up queries the user might find useful,
formatted exactly as a numbered list.
```

---

## 🏎 Performance tips

* **Chunk size 1000 / overlap 100** is a sweet‑spot for annual reports.
* Want faster? Swap `sentence-transformers/all-MiniLM-L6-v2` → `bge-small-en`.
* Need citations? Use `return_source_documents=True` on the chain.

---



