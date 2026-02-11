# 📚 DocRAG — Document Q&A with RAG

A containerized **Retrieval Augmented Generation** system that lets you upload documents and ask questions about them. Built with **FastAPI**, **Google Gemini**, **ChromaDB**, and **Streamlit**.

## Features

- **Upload PDFs & Text files** — documents are chunked and embedded locally
- **Ask questions** — answers are generated using your uploaded documents as context
- **Streaming responses** — answers appear in real-time
- **Fully containerized** — one command to run everything with Docker

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| **Vector Store** | ChromaDB |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Infra** | Docker Compose |

## Quick Start

### 1. Get a Google API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your Google API key
```

### 3. Run with Docker

```bash
docker compose up --build
```

### 4. Use

- **Frontend**: [http://localhost:8501](http://localhost:8501)
- **Backend API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

Upload a document in the sidebar, then ask questions in the chat.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload PDF/TXT files |
| `POST` | `/api/chat` | Ask a question (streaming SSE) |

## Project Structure

```
doc_rag/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py      # API routes
│   │   ├── core/config.py        # Settings
│   │   └── services/
│   │       ├── ingestion.py      # Document processing
│   │       ├── rag_chain.py      # LLM + retrieval
│   │       └── vector_store.py   # ChromaDB
│   ├── tests/test_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                    # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## Running Without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Set `GOOGLE_API_KEY` as an environment variable, or enter it in the frontend sidebar.
