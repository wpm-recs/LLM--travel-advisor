# Global Travel Advisor (RAG)

A Retrieval-Augmented Generation (RAG) project for global travel Q&A, backed by local Wikivoyage markdown files.

This repository now includes:
- CLI chat mode (existing)
- A complete browser frontend (new)
- A lightweight Python web server API (new)

## Features

- Hybrid retrieval: FAISS dense retrieval + BM25 sparse retrieval
- Metadata-aware chunking from markdown headers
- Moonshot/Kimi LLM answer generation
- Beautiful responsive frontend (desktop + mobile)
- Zero extra web framework required for frontend serving

## Project Structure

- `main.py`: CLI entry and core `TravelRAGSystem`
- `web_app.py`: Web server (`/api/health`, `/api/ask`) + static frontend hosting
- `frontend/index.html`: UI markup
- `frontend/styles.css`: Visual design and responsive styles
- `frontend/app.js`: Frontend logic and API integration
- `rag_modules/`: Data prep, indexing, retrieval, generation
- `wikivoyage_global/`: Global knowledge base source (recommended)
- `wikivoyage_sg/`: Backward-compatible fallback source

## Requirements

- Python 3.10+
- `MOONSHOT_API_KEY` environment variable

## Installation

```bash
conda create -n travel-advisor python=3.10 -y
conda activate travel-advisor
pip install -r requirements.txt
```

Set your API key:

```bash
export MOONSHOT_API_KEY="your_api_key_here"
```

Optional embedding device override (default is CPU):

```bash
export EMBEDDING_DEVICE="cpu"
```

Optional data path override:

```bash
export TRAVEL_DATA_PATH="./wikivoyage_global"
```

## Run (Web Frontend)

Start server:

```bash
python web_app.py
```

Open in browser:

- `http://127.0.0.1:8080`

Optional host/port customization:

```bash
export TRAVEL_WEB_HOST="0.0.0.0"
export TRAVEL_WEB_PORT="8080"
python web_app.py
```

## Run (CLI)

```bash
python main.py
```

## API Endpoints

- `GET /api/health`
	- Returns backend readiness status
- `POST /api/ask`
	- JSON body: `{"question": "your travel question"}`
	- JSON response: `{"answer": "...", "question": "..."}`

## First-Run Notes

- First run is slower because embeddings/index are built.
- Index and chunks are cached locally for later runs:
	- `saved_chunks_global.pkl`
	- `vector_index_global/`

## Troubleshooting

- `数据路径不存在` / data path errors:
	- Ensure `wikivoyage_global/` exists in repo root, or set `TRAVEL_DATA_PATH`.
	- If global data is absent, the system automatically falls back to `wikivoyage_sg/` when available.
- API key errors:
	- Ensure `MOONSHOT_API_KEY` is exported in the same shell.
- Slow startup:
	- Expected on first run while building vector index.