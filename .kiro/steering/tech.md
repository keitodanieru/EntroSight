# EntroSight — Tech Stack & Build

## Language

- Python 3.11

## Framework & Libraries

| Category | Library |
|----------|---------|
| Web framework | FastAPI + Uvicorn |
| Templating | Jinja2 |
| Frontend interactivity | HTMX (CDN) |
| ML inference | PyTorch (CPU), torchvision |
| Image processing | Pillow, matplotlib, numpy |
| Vector DB / RAG | ChromaDB (PersistentClient) |
| LLM | Ollama (Mistral model, via HTTP API) |
| HTTP client | httpx (async) |
| Database | SQLite via aiosqlite |
| Config | pydantic-settings (env prefix: `ENTROSIGHT_`) |
| File uploads | python-multipart |
| Testing | pytest, pytest-asyncio, hypothesis |

## Deployment

- Docker Compose with two services:
  - `app` — Python application container (port 8000)
  - `ollama` — Ollama LLM container (port 11434)
- Base image: Python 3.11 slim
- CPU-only PyTorch (no GPU required)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Run with Docker Compose
docker-compose up --build

# Run specific test file
pytest tests/test_validator.py -v
```

## Environment Variables

All settings use the `ENTROSIGHT_` prefix. Key variables:

- `ENTROSIGHT_MODEL_CHECKPOINT_PATH` — path to .pth model file
- `ENTROSIGHT_OLLAMA_BASE_URL` — Ollama service URL (default: http://ollama:11434)
- `ENTROSIGHT_OLLAMA_MODEL` — LLM model name (default: mistral)
- `ENTROSIGHT_DATABASE_PATH` — SQLite DB path (default: data/scans.db)
- `ENTROSIGHT_CHROMADB_PATH` — ChromaDB storage path (default: data/chromadb)
