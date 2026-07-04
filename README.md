# EntroSight

A privacy-preserving Windows PE malware family classifier with explainable threat intelligence.

## What It Does

EntroSight accepts uploaded PE binaries (.exe, .dll, .sys), converts them to byte-entropy heatmaps, classifies them into malware families using a fine-tuned ResNet50 model, and generates plain-language threat explanations backed by MITRE ATT&CK intelligence.

**Supported families:** AgentTesla, Remcos, DCRat, AsyncRAT, RedLineStealer, Formbook, and Benign.

## Key Principles

- **Privacy-preserving** — uploaded binaries are deleted from memory immediately after feature extraction. No data leaves your environment.
- **Explainable AI** — classifications come with MITRE ATT&CK-grounded explanations, not just labels.
- **Fully local** — all inference, retrieval, and LLM generation run on-premises via Docker Compose.

## Architecture

```
User Browser → Web UI (Jinja2 + HTMX) → FastAPI Backend
    → File Validator → Entropy Heatmap Generator → ResNet50 Classifier
    → RAG Engine (ChromaDB + MITRE ATT&CK) → Ollama LLM → Explanation
    → SQLite (scan history)
```

Two Docker containers:
- `app` — Python application (port 8000)
- `ollama` — Ollama LLM service (port 11434)

## Requirements

- Python 3.11+
- Docker & Docker Compose (for deployment)
- A trained ResNet50 checkpoint file (`.pth`) placed in `models/`

## Quick Start (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

## Quick Start (Docker)

```bash
# Build and run both containers
docker-compose up --build

# The app will be available at http://localhost:8000
# Ollama pulls the Mistral model on first startup
```

## Environment Variables

All settings use the `ENTROSIGHT_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTROSIGHT_MODEL_CHECKPOINT_PATH` | `models/resnet50_malware.pth` | Path to the trained model |
| `ENTROSIGHT_OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama service URL |
| `ENTROSIGHT_OLLAMA_MODEL` | `mistral` | LLM model for explanations |
| `ENTROSIGHT_DATABASE_PATH` | `data/scans.db` | SQLite database path |
| `ENTROSIGHT_CHROMADB_PATH` | `data/chromadb` | ChromaDB storage path |
| `ENTROSIGHT_MAX_FILE_SIZE_MB` | `50` | Max upload size in MB |

## Project Structure

```
entrovision/
├── app/
│   ├── main.py                 # FastAPI app, routes, lifespan
│   ├── config.py               # AppSettings (pydantic-settings)
│   ├── models.py               # Pydantic/dataclass models
│   ├── scan.py                 # Scan orchestration pipeline
│   ├── knowledge_base_loader.py
│   ├── components/
│   │   ├── validator.py        # PE file validation
│   │   ├── heatmap.py          # Byte-entropy heatmap generation
│   │   ├── classifier.py       # ResNet50 inference
│   │   ├── rag.py              # ChromaDB MITRE ATT&CK retrieval
│   │   ├── explainer.py        # Ollama LLM explanations
│   │   └── database.py         # Async SQLite scan history
│   ├── templates/              # Jinja2 HTML templates
│   └── static/css/             # Stylesheets
├── data/
│   └── knowledge_base/         # MITRE ATT&CK JSON documents
├── models/                     # ML checkpoints (.pth)
├── tests/                      # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Checkpoint

The `.pth` checkpoint is trained separately and not included in this repo. Place it at `models/resnet50_malware.pth` (or configure via env var). The checkpoint must contain a `"model_state_dict"` key with weights for a ResNet50 with 7-class output.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific component
pytest tests/test_validator.py -v
pytest tests/test_classifier.py -v
pytest tests/test_rag.py -v

# Property-based tests
pytest tests/test_heatmap_properties.py -v
```

## Target Performance

- Classification: < 1 second (CPU)
- Full scan-to-explanation pipeline: < 30 seconds
