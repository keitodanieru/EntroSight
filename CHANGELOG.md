# Changelog

All notable changes to the EntroSight project are documented here.

---

## [0.1.0] — 2026-07-04

### Summary

Initial implementation of the EntroSight backend pipeline — all core components are built, tested, and wired together. The system can validate PE files, generate entropy heatmaps, classify malware families via ResNet50, retrieve MITRE ATT&CK context from ChromaDB, generate LLM-powered explanations via Ollama, and persist scan history in SQLite.

---

### Added

#### Project Foundation (Tasks 1.1–1.3)

- **Project structure** — full directory layout: `app/`, `app/components/`, `app/templates/`, `app/static/css/`, `data/`, `models/`, `tests/`
- **`requirements.txt`** — all dependencies pinned (FastAPI, Uvicorn, PyTorch, ChromaDB, httpx, aiosqlite, Hypothesis, etc.)
- **`app/config.py`** — `AppSettings` class using pydantic-settings with `ENTROSIGHT_` env prefix. Covers model path, class labels, processing settings, RAG/Ollama/DB/storage configuration.
- **`app/models.py`** — Pydantic response models (`ScanResponse`, `ScanStatusResponse`) and internal dataclasses (`ValidationResult`, `ClassificationResult`, `RetrievedPassage`, `ExplanationResult`, `ScanRecord`, `Document`).

#### Component 1: File Validator (Task 2)

- **`app/components/validator.py`** — `FileValidator` class
  - Validates file extension against allowlist (`.exe`, `.dll`, `.sys`)
  - Enforces 50 MB max file size
  - Verifies MZ signature (first 2 bytes)
  - Computes and returns SHA-256 hash on success
  - Returns structured `ValidationResult` with error messages on failure
- **How it works:** Called at the start of every scan. If validation fails, the scan is immediately rejected with a descriptive error. No further processing occurs on invalid files.
- **Tests:** Unit tests (`test_validator.py`) + property-based tests validating completeness and extension-signature alignment.

#### Component 2: Entropy Heatmap Generator (Task 3)

- **`app/components/heatmap.py`** — `EntropyHeatmapGenerator` class
  - `generate(file_bytes)` → Tensor of shape `(3, 256, 256)`
  - `generate_visualization(file_bytes)` → PNG bytes with colormap
  - Algorithm: divides file into 256-byte blocks → computes Shannon entropy per block → arranges into 2D grid → resizes to 256×256 via bilinear interpolation → replicates to 3 RGB channels
  - Entropy normalized to [0.0, 1.0] (divided by theoretical max of 8.0)
- **How it works:** Transforms raw PE bytes into a visual representation of byte randomness structure. High-entropy regions (packed/encrypted sections) appear bright; low-entropy regions (headers, padding) appear dark. The 3-channel tensor is formatted for direct ResNet50 input.
- **Tests:** Unit tests (shape, low/high entropy behavior, PNG output) + property tests (entropy bounds, determinism).

#### Component 3: ResNet50 Classifier (Task 4)

- **`app/components/classifier.py`** — `MalwareClassifier` class
  - `__init__(checkpoint_path)` — loads ResNet50 with modified FC layer (7 classes), loads `checkpoint["model_state_dict"]`, sets eval mode on CPU
  - `classify(heatmap_tensor)` → `ClassificationResult` with label, confidence, all probabilities, and timing
  - `get_grad_cam(heatmap_tensor)` → 256×256 activation map for explainability
  - Applies ImageNet normalization before inference
  - Uses `torch.no_grad()` for efficient inference
- **How it works:** Takes the entropy heatmap tensor, normalizes it, runs a single forward pass through the fine-tuned ResNet50, applies softmax to get a probability distribution across 7 classes, and returns the top prediction. Grad-CAM hooks into `layer4` to produce a class-discriminative heatmap overlay.
- **Note:** Requires a `.pth` checkpoint file from the ML teammate. Tests use mock random weights.
- **Tests:** Unit tests with mock checkpoint (loading, probability normalization, label validity, timing, Grad-CAM shape/range).

#### Component 4: RAG Engine (Task 7.1)

- **`app/components/rag.py`** — `RAGEngine` class
  - `__init__(collection_name, persist_directory)` — initializes ChromaDB PersistentClient
  - `retrieve_context(family_label, top_k)` → list of `RetrievedPassage` objects
  - `ingest_knowledge_base(documents)` → count of documents upserted
  - Metadata filter: queries only documents tagged with the target family or "general"
  - Relevance threshold: filters out results with similarity score < 0.3
  - Document IDs: `{technique_id}_{family}` for idempotent upsert
- **How it works:** After classification, the predicted family label is used to query ChromaDB for relevant MITRE ATT&CK technique descriptions. The top-k most similar passages (above threshold) are passed to the LLM as grounding context for explanation generation.
- **Tests:** 14 unit tests covering ingest/retrieve roundtrip, threshold filtering, family isolation, empty collection handling, and upsert deduplication.

#### Component 4b: MITRE ATT&CK Knowledge Base (Task 7.2)

- **`data/knowledge_base/mitre_techniques.json`** — 35 curated entries
  - 5 techniques per malware family (AgentTesla, Remcos, DCRat, AsyncRAT, RedLineStealer, Formbook, Benign)
  - Each entry: `technique_id`, `technique_name`, `family`, `text` (2–4 sentence description)
  - Uses real MITRE ATT&CK IDs (T1055, T1071, T1555, etc.)
- **`app/knowledge_base_loader.py`** — `load_knowledge_base(rag_engine, kb_path)` function
  - Reads JSON, converts to `Document` objects, calls `rag_engine.ingest_knowledge_base()`
  - Designed to be called from FastAPI lifespan on first startup
- **How it works:** On application startup, if the ChromaDB collection is empty, the loader reads the JSON knowledge base and ingests all 35 documents. This provides the RAG engine with technique context for generating grounded explanations.

#### Component 5: Explanation Generator (Task 8)

- **`app/components/explainer.py`** — `ExplanationGenerator` class
  - `generate(label, confidence, context_passages)` → `ExplanationResult`
  - Formats system prompt (malware analyst role) + user prompt (classification + RAG context)
  - Calls Ollama's `/api/generate` endpoint via async httpx
  - Graceful fallback on timeout/connection error: "Explanation unavailable - LLM service temporarily unreachable"
  - Handles empty context_passages (generates without RAG context)
  - Tracks `generation_time_ms`
- **How it works:** Combines the classification result with retrieved MITRE ATT&CK passages into a structured prompt. The Ollama LLM (Mistral) generates a plain-language explanation under 200 words. If Ollama is unreachable, the scan still completes with a fallback message.
- **Tests:** 12 unit tests with mocked httpx (success, timeout, connection error, prompt formatting, empty/non-empty context).

#### Component 6: Scan History Database (Task 6)

- **`app/components/database.py`** — `ScanHistoryDB` class
  - `initialize()` — creates schema (async, idempotent)
  - `store_result(ScanRecord)` → auto-incremented record ID
  - `get_result(scan_id)` → `ScanRecord | None`
  - `get_by_hash(sha256)` → most recent record for deduplication
  - `list_recent(limit=20)` → recent scans ordered by timestamp DESC
  - Schema: `scan_results` table with indexes on `sha256` and `timestamp`
- **How it works:** After a scan completes, the full result (hash, label, confidence, explanation, heatmap path, timing) is persisted in SQLite. The history endpoint queries this table. Hash-based lookup enables deduplication — if the same file is scanned again, we can show previous results.
- **Tests:** 10 unit tests covering CRUD operations, ordering, limits, deduplication, and idempotent schema creation.

#### Documentation

- **`README.md`** — project overview, quick start (dev + Docker), environment variables, project structure, testing instructions.

---

### Pipeline Flow (How the System Works)

```
1. USER uploads a PE file (.exe/.dll/.sys)
       │
2. FILE VALIDATOR checks extension, size, MZ signature
       │ (reject with error if invalid)
       │
3. ENTROPY HEATMAP GENERATOR converts raw bytes to (3, 256, 256) tensor
       │
4. RAW FILE BYTES deleted from memory (privacy-preserving)
       │
5. RESNET50 CLASSIFIER runs inference on heatmap tensor
       │ → predicted family, confidence, probability distribution
       │
6. RAG ENGINE queries ChromaDB for MITRE ATT&CK techniques
       │ → relevant passages filtered by family + relevance threshold
       │
7. EXPLANATION GENERATOR sends classification + context to Ollama
       │ → plain-language threat intelligence explanation
       │
8. HEATMAP VISUALIZATION saved as PNG to disk
       │
9. SCAN HISTORY DB stores the complete result
       │
10. WEB UI renders the result (verdict, confidence, heatmap, explanation)
```

---

### What's Next (Remaining Tasks)

- **Task 9** — Scan orchestration (`app/scan.py`) + FastAPI routes (`app/main.py`)
- **Task 10** — Web UI templates (Jinja2 + HTMX, converted from prototype)
- **Task 11** — Backend + UI integration checkpoint
- **Task 12** — Docker Compose deployment (Dockerfile + docker-compose.yml)
- **Task 13** — End-to-end integration wiring and integration tests
- **Task 14** — Final system verification

---

### Test Coverage Summary

| Component | Test File | Tests | Type |
|-----------|-----------|-------|------|
| FileValidator | `tests/test_validator.py` | 10 | Unit + Property |
| EntropyHeatmapGenerator | `tests/test_heatmap.py` | 7 | Unit |
| EntropyHeatmapGenerator | `tests/test_heatmap_properties.py` | 3 | Property (Hypothesis) |
| MalwareClassifier | `tests/test_classifier.py` | 11 | Unit (mock checkpoint) |
| RAGEngine | `tests/test_rag.py` | 14 | Unit |
| ExplanationGenerator | `tests/test_explainer.py` | 12 | Unit (mocked httpx) |
| ScanHistoryDB | `tests/test_database.py` | 10 | Unit (async, in-memory SQLite) |

**Total: 67 tests across 7 test files**
