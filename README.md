# 📚 DocRAG — Production-Grade Document Q&A with Hybrid RAG

A **Retrieval-Augmented Generation (RAG)** system featuring **Two-Stage Hybrid Search (Dense + BM25 Sparse)**, **Cross-Encoder Reranking**, and rigorous **Ragas Benchmark Evaluation**. Built with **FastAPI**, **Google Gemini**, **ChromaDB**, **Groq**, and **Streamlit**.

---

## 🚀 Key Features

- **Two-Stage Retrieval Pipeline**:
  - **Stage 1 (High Recall)**: Hybrid retrieval combining dense vector similarity (`all-MiniLM-L6-v2`) with sparse keyword matching (`BM25`).
  - **Stage 2 (High Precision)**: Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) that scores query-document pairs to surface the most relevant chunks.
- **Document Ingestion**: Upload PDFs & text files with automatic text chunking and vector/keyword indexing.
- **Streaming Q&A**: Real-time response streaming using Google Gemini and FastAPI SSE (Server-Sent Events).
- **Ragas Evaluation Benchmark**: Automated evaluation framework benchmarking Faithfulness, Answer Relevancy, Context Precision, and Context Recall using an independent LLM judge on Groq.

---

## 📊 Evaluation Results

We benchmarked the upgraded retrieval pipeline (hybrid dense+BM25 search with cross-encoder reranking) against the original dense-only baseline using [Ragas](https://github.com/explodinggradients/ragas), with an independent LLM judge (`Groq / openai/gpt-oss-120b`) to avoid same-model bias between generator and evaluator.

| Metric | Baseline (Dense-only $k=4$) | Upgraded (Hybrid $k=10$ + Rerank top-$4$) | Change |
|:---|:---:|:---:|:---:|
| **Context Precision** | `0.80` | `0.99` | **+23.1%** |
| **Answer Relevancy** | `0.73` | `0.74` | **+1.8%** |
| **Faithfulness** | `0.83` | `0.83` | **+0.0%** |
| **Context Recall** | `0.83` | `0.83` | **+0.0%** |

*Evaluated on a benchmark across multi-domain document QA datasets.*

> **Key Finding:** Hybrid retrieval + Cross-Encoder reranking delivered its largest gains in **Context Precision (+23.1%)** — meaning retrieved chunks were substantially less noisy and placed the most relevant source facts at the top of the context window. Faithfulness and Context Recall remained stable and high without introducing hallucinations.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **LLM Generator** | Google Gemini (`gemini-2.5-flash-lite` / `gemini-3.5-flash-lite`) | Answers user questions based on context |
| **Dense Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Runs locally for fast vector search |
| **Sparse Keyword Search**| `rank-bm25` | Lexical recall for exact keyword & term matches |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Joint query-document relevance scoring |
| **Vector Database** | ChromaDB | Local persistent vector storage |
| **Eval Judge LLM** | Groq (`openai/gpt-oss-120b`) | Independent evaluation judge via Ragas |
| **Backend** | FastAPI + Uvicorn | High-performance asynchronous REST API |
| **Frontend** | Streamlit | Interactive web UI with real-time streaming |

---

## 🏗️ Architecture

```
[User Document (PDF/TXT)]
           │
           ▼
[Chunking & Ingestion] ───────────────┬────────────────────────┐
                                      ▼                        ▼
                             [ChromaDB VectorStore]    [BM25 Sparse Corpus]
                                      │                        │
[User Query] ─────────────────────────┴────────┬───────────────┘
                                               ▼
                              [Hybrid Search Candidates (k=10)]
                                               │
                                               ▼
                              [Cross-Encoder Reranker (top-4)]
                                               │
                                               ▼
                                 [Prompt Context Assembly]
                                               │
                                               ▼
                                [Google Gemini Generator]
                                               │
                                               ▼
                                [Streaming Answer Output]
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.11+
- [Google AI Studio API Key](https://aistudio.google.com/apikey) (for RAG Answer Generation)
- [Groq API Key](https://console.groq.com/keys) (Optional, for running Ragas Evaluation benchmarks)

### 2. Environment Setup

Clone the repository and create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install & Run

#### Backend
```bash
cd backend
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

#### Frontend (separate terminal)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

- **Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📈 Running the Ragas Benchmark Evaluation

To execute the end-to-end evaluation pipeline comparing Baseline vs. Upgraded RAG:

```bash
cd backend
python -m eval.run_evaluation
```

To regenerate the markdown comparison report:
```bash
python -m eval.generate_report
```

*Results are saved to `backend/eval/results/comparison_report.md`.*

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status check |
| `POST` | `/api/upload` | Upload and index PDF / TXT documents |
| `POST` | `/api/chat` | Ask questions with Server-Sent Events (SSE) streaming |

---

## 📂 Project Structure

```
doc_rag/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py       # FastAPI routing & SSE streaming
│   │   ├── core/config.py         # Dynamic model discovery & configuration
│   │   └── services/
│   │       ├── ingestion.py       # PDF/TXT parser & text chunker
│   │       ├── rag_chain.py       # Hybrid retrieval & Gemini generation
│   │       ├── reranker.py        # Cross-Encoder precision reranker
│   │       └── vector_store.py    # ChromaDB & BM25 sparse index
│   ├── eval/
│   │   ├── qa_dataset.json        # Benchmark questions & ground truth references
│   │   ├── ragas_setup.py         # Groq judge & rate-limited Ragas configuration
│   │   ├── run_evaluation.py      # Benchmark runner (Baseline vs Upgraded)
│   │   ├── generate_report.py     # Comparison report generator
│   │   └── results/               # Evaluation outputs (.json & .md)
│   ├── tests/test_api.py
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Streamlit chat interface
│   └── requirements.txt
├── .env.example
└── README.md
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
