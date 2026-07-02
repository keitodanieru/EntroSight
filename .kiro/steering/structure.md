# EntroSight — Project Structure

```
entrovision/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, routes, lifespan handler
│   ├── config.py                 # AppSettings (pydantic-settings)
│   ├── models.py                 # Pydantic request/response models
│   ├── scan.py                   # Scan orchestration background task
│   ├── components/               # Core pipeline components
│   │   ├── __init__.py
│   │   ├── validator.py          # FileValidator — PE file validation
│   │   ├── heatmap.py            # EntropyHeatmapGenerator — byte-entropy conversion
│   │   ├── classifier.py         # MalwareClassifier — ResNet50 inference
│   │   ├── rag.py                # RAGEngine — ChromaDB retrieval
│   │   ├── explainer.py          # ExplanationGenerator — Ollama LLM calls
│   │   └── database.py           # ScanHistoryDB — async SQLite
│   ├── templates/                # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── index.html            # Upload page
│   │   ├── result.html           # Scan result page
│   │   ├── history.html          # Scan history page
│   │   └── partials/             # HTMX partial fragments
│   │       ├── scan_status.html
│   │       └── result_card.html
│   └── static/
│       └── css/
│           └── style.css
├── data/
│   ├── knowledge_base/           # MITRE ATT&CK JSON documents for RAG ingestion
│   ├── chromadb/                 # ChromaDB persistent storage (generated)
│   ├── heatmaps/                 # Saved heatmap PNGs (generated)
│   └── scans.db                  # SQLite scan history (generated)
├── models/                       # ML model checkpoints (.pth files from teammate)
├── tests/                        # pytest test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Key Conventions

- **Components pattern**: Each pipeline stage is a class in `app/components/`. They are initialized in the FastAPI lifespan and attached to `app.state`.
- **Async everywhere**: Database and HTTP calls use async (aiosqlite, httpx). The scan pipeline runs as a FastAPI `BackgroundTask`.
- **Templates + HTMX**: Server-rendered HTML with HTMX for dynamic updates (polling scan status, form submissions). No JS framework.
- **Generated data**: `data/chromadb/`, `data/heatmaps/`, and `data/scans.db` are created at runtime. Don't commit them.
- **Model checkpoint**: The `.pth` file in `models/` comes from a teammate. The codebase only wraps inference, not training.
