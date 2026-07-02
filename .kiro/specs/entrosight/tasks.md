# Implementation Plan: EntroSight

## Overview

Implementation plan for the EntroSight malware family classifier backend and web UI. This covers the Dev (Member B) scope: FastAPI backend, inference pipeline, RAG engine, web UI, scan history database, and Docker Compose deployment. ML model training is handled separately by a teammate — the Dev receives a `.pth` checkpoint file.

## Tasks

- [-] 1. Project structure and core configuration
  - [x] 1.1 Initialize project directory structure and dependencies
    - Create the full project layout: `app/`, `app/components/`, `app/templates/`, `app/templates/partials/`, `app/static/css/`, `data/`, `data/knowledge_base/`, `models/`, `tests/`
    - Create `requirements.txt` with all dependencies (fastapi, uvicorn, jinja2, python-multipart, torch, torchvision, chromadb, httpx, pydantic, pydantic-settings, aiosqlite, pillow, matplotlib, numpy, hypothesis, pytest, pytest-asyncio)
    - Create `app/__init__.py` and `app/components/__init__.py`
    - _Requirements: Project structure from design document_

  - [x] 1.2 Implement application configuration module
    - Create `app/config.py` with `AppSettings` class using pydantic-settings
    - Include all settings: model_checkpoint_path, class_labels, max_file_size_mb, entropy_block_size, heatmap_image_size, chromadb_path, rag_collection_name, rag_top_k, ollama_base_url, ollama_model, explanation_max_tokens, ollama_timeout_seconds, database_path, heatmap_storage_dir
    - Use `ENTROSIGHT_` env prefix for all settings
    - _Requirements: AppSettings from design data models_

  - [x] 1.3 Define Pydantic request/response models
    - Create `app/models.py` with `ScanResponse`, `ScanStatusResponse`, `ValidationResult`, `ClassificationResult`, `RetrievedPassage`, `ExplanationResult`, `ScanRecord` dataclasses/models
    - Ensure all models match the design interfaces exactly
    - _Requirements: Data Models section of design_

- [ ] 2. File validation component
  - [~] 2.1 Implement FileValidator class
    - Create `app/components/validator.py` with `FileValidator` class
    - Implement extension checking against allowlist (.exe, .dll, .sys)
    - Implement file size enforcement (50 MB max)
    - Implement MZ signature verification (first 2 bytes == b"MZ")
    - Compute and return SHA-256 hash on valid files
    - _Requirements: Component 1 design, Property 2 (Validation Completeness), Property 10 (Extension-Signature Alignment)_

  - [ ]* 2.2 Write property tests for FileValidator
    - **Property 2: Validation Completeness** — for all inputs, either validation passes and file proceeds, or it fails with an error and no processing occurs
    - **Property 10: Extension-Signature Alignment** — for all files that pass, they have both valid extension AND valid MZ signature
    - **Validates: Properties 2, 10 from design**

  - [ ]* 2.3 Write unit tests for FileValidator
    - Test valid PE file with correct extension, size, and MZ header
    - Test invalid extension (.txt, .pdf, no extension)
    - Test oversized file (> 50 MB)
    - Test file missing MZ signature
    - Test empty file
    - _Requirements: Component 1 design, Testing Strategy_

- [ ] 3. Entropy heatmap generator
  - [~] 3.1 Implement EntropyHeatmapGenerator class
    - Create `app/components/heatmap.py` with `EntropyHeatmapGenerator` class
    - Implement `generate()` method: divide file into 256-byte blocks, compute Shannon entropy per block, arrange into 2D grid, resize to 256×256 via bilinear interpolation, replicate to 3 RGB channels
    - Implement `generate_visualization()` method: apply matplotlib colormap, return PNG bytes using Pillow
    - Normalize entropy values to [0, 1] range (divide by 8.0)
    - _Requirements: Component 2 design, Algorithm 1 (Byte-Entropy Heatmap Generation), Property 5 (Entropy Bounds)_

  - [ ]* 3.2 Write property tests for EntropyHeatmapGenerator
    - **Property 5: Entropy Bounds** — for any byte sequence, computed Shannon entropy is in [0.0, 8.0] and normalized value in [0.0, 1.0]
    - **Property 3: Classification Determinism (heatmap portion)** — same input bytes always produce same output tensor
    - **Validates: Properties 3, 5 from design**

  - [ ]* 3.3 Write unit tests for EntropyHeatmapGenerator
    - Test output tensor shape is (3, 256, 256)
    - Test all-zero bytes produce low entropy
    - Test random bytes produce high entropy
    - Test visualization returns valid PNG bytes
    - _Requirements: Component 2 design, Testing Strategy_

- [ ] 4. Model loading and inference wrapper
  - [~] 4.1 Implement MalwareClassifier class
    - Create `app/components/classifier.py` with `MalwareClassifier` class
    - Implement `__init__()`: load ResNet50 architecture, modify final FC layer to match num_classes (7), load checkpoint state_dict, set eval mode, move to CPU
    - Implement `classify()`: apply ImageNet normalization, run forward pass with torch.no_grad(), softmax for probabilities, extract top prediction with confidence and timing
    - Implement `get_grad_cam()`: generate Grad-CAM activation map from last conv layer for explainability overlay
    - Handle checkpoint key "model_state_dict" as per design
    - _Requirements: Component 3 design, Algorithm 2 (Classification Pipeline), Property 4 (Probability Normalization)_

  - [ ]* 4.2 Write unit tests for MalwareClassifier
    - Mock a checkpoint file with random weights to test loading logic
    - Test classify() returns valid ClassificationResult with probabilities summing to ~1.0
    - Test output label is one of CLASS_LABELS
    - Test inference_time_ms is positive
    - _Requirements: Component 3 design, Property 4_

- [~] 5. Checkpoint — Ensure all core components pass tests
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. SQLite scan history database
  - [~] 6.1 Implement ScanHistoryDB class
    - Create `app/components/database.py` with `ScanHistoryDB` class
    - Implement schema creation on init (scan_results table with indexes on sha256 and timestamp)
    - Implement `store_result()`: insert ScanRecord, return generated ID
    - Implement `get_result()`: retrieve by scan ID
    - Implement `get_by_hash()`: lookup by SHA-256 for deduplication
    - Implement `list_recent()`: return most recent N scans ordered by timestamp DESC
    - Use aiosqlite for async access
    - _Requirements: Component 6 design, Database Schema model, Property 7 (Scan Completeness)_

  - [ ]* 6.2 Write unit tests for ScanHistoryDB
    - Test schema creation with in-memory SQLite
    - Test store and retrieve roundtrip
    - Test get_by_hash returns correct record
    - Test list_recent respects limit and ordering
    - Test duplicate hash detection
    - _Requirements: Component 6 design, Property 6 (Hash Integrity), Property 7_

- [ ] 7. ChromaDB RAG engine and MITRE ATT&CK knowledge base
  - [~] 7.1 Implement RAGEngine class
    - Create `app/components/rag.py` with `RAGEngine` class
    - Implement `__init__()`: initialize ChromaDB PersistentClient, get_or_create collection "mitre_attack"
    - Implement `retrieve_context()`: format query with family label, query ChromaDB with metadata filter, convert distances to similarity scores, filter by 0.3 threshold, return ranked RetrievedPassage list
    - Implement `ingest_knowledge_base()`: accept list of Document objects, upsert into ChromaDB collection with metadata (technique_id, technique_name, family)
    - _Requirements: Component 4 design, Algorithm 3 (RAG Context Retrieval)_

  - [~] 7.2 Curate MITRE ATT&CK knowledge base documents
    - Create `data/knowledge_base/` with JSON/JSONL documents mapping malware families to MITRE ATT&CK techniques
    - Cover all 7 class labels: AgentTesla, Remcos, DCRat, AsyncRAT, RedLineStealer, Formbook, Benign
    - Each document: technique_id (e.g., T1055), technique_name, description text, associated family metadata
    - Create a startup ingestion script/function that loads knowledge base into ChromaDB on first run
    - _Requirements: Component 4 design, RAG Engine responsibilities_

  - [ ]* 7.3 Write unit tests for RAGEngine
    - Test ingest and retrieve roundtrip with mock documents
    - Test relevance threshold filtering (documents below 0.3 excluded)
    - Test retrieve returns empty list for unknown family
    - _Requirements: Component 4 design_

- [ ] 8. Ollama explanation generator
  - [~] 8.1 Implement ExplanationGenerator class
    - Create `app/components/explainer.py` with `ExplanationGenerator` class
    - Implement `generate()`: format system prompt (malware analyst role), build user prompt with classification result + RAG context passages, call Ollama `/api/generate` endpoint via httpx with configured timeout, parse response, return ExplanationResult with timing
    - Handle Ollama timeout with graceful fallback message: "Explanation unavailable - LLM service temporarily unreachable"
    - Handle empty context_passages gracefully (generate without RAG context)
    - _Requirements: Component 5 design, Algorithm 4 (Explanation Generation), Error Scenario 3_

  - [ ]* 8.2 Write unit tests for ExplanationGenerator
    - Mock httpx responses to test prompt formatting
    - Test timeout handling returns fallback message
    - Test generation_time_ms is tracked correctly
    - _Requirements: Component 5 design, Error Scenario 3_

- [ ] 9. FastAPI backend and scan orchestration
  - [~] 9.1 Implement scan orchestration background task
    - Create `app/scan.py` with `execute_scan()` function
    - Implement the full pipeline: validate → generate heatmap → delete file bytes → classify → retrieve context → generate explanation → save heatmap PNG → store in DB
    - Implement scan status tracking (in-memory dict with states: pending, processing, complete, error)
    - Implement `update_scan_status()` helper with progress_stage tracking (validating, classifying, explaining)
    - Handle all error scenarios: validation failure, memory error, DB write failure
    - _Requirements: Algorithm 5 (Full Scan Orchestration), Property 1 (File Privacy), Property 7 (Scan Completeness)_

  - [~] 9.2 Implement FastAPI application with routes and lifespan
    - Create `app/main.py` with FastAPI app instance
    - Implement lifespan handler: initialize all components (validator, heatmap generator, classifier, RAG engine, explainer, DB), load model checkpoint, ingest knowledge base if ChromaDB empty
    - Implement routes:
      - `POST /api/scan` — accept multipart file upload, launch background task, return 202 with scan_id
      - `GET /api/scan/{scan_id}/status` — return current scan status for HTMX polling
      - `GET /` — render upload page
      - `GET /history` — render scan history page
      - `GET /scan/{scan_id}` — render individual scan result page
    - Mount static files directory
    - Configure Jinja2 templates
    - _Requirements: Example Usage from design, main scan flow sequence diagram_

- [ ] 10. Web UI templates (Jinja2 + HTMX) — convert from `.dc.html` prototype
  - [ ] 10.1 Create base template and CSS from prototype
    - Convert `PE Malware Classification Platform/Entropy - PE Malware Classifier.dc.html` into Jinja2 templates
    - Create `app/templates/base.html` — extract header (logo SVG, nav with Scan/History tabs, "Local · offline" badge), footer, Google Fonts links (Playfair Display, DM Sans, IBM Plex Mono), HTMX script (CDN)
    - Create `app/static/css/style.css` — extract all inline styles from the prototype into a proper stylesheet. Dark theme (#0e1113 bg), accent red (#e5484d), monospace labels (IBM Plex Mono), serif headings (Playfair Display)
    - Preserve the design system: color palette, typography scale, spacing, grid layouts, animations (pulse, shimmer, scan, fadein)
    - _Source: `.dc.html` header/footer/style sections_

  - [~] 10.2 Create upload page (index.html)
    - Convert the `<sc-if value="{{ isUpload }}">` section into `app/templates/index.html` extending base
    - Preserve: hero heading, description, drop-zone with drag/drop styling, file input accepting .exe/.dll/.sys
    - Wire upload to HTMX: `hx-post="/api/scan"`, `hx-encoding="multipart/form-data"`, `hx-target="#result-container"`
    - Include the 5-step pipeline grid (Validate → Visualize → Classify → Contextualize → Explain)
    - _Source: `.dc.html` UPLOAD section_

  - [~] 10.3 Create analyzing/progress partial
    - Convert the `<sc-if value="{{ isAnalyzing }}">` section into `app/templates/partials/scan_status.html`
    - Preserve: file name display, shimmer animation placeholder, scan-line animation, step progress list with done/active/pending states
    - Wire HTMX polling: `hx-get="/api/scan/{scan_id}/status"`, `hx-trigger="every 2s"`, `hx-swap="outerHTML"`
    - Map progress_stage values (validating, classifying, explaining) to step indicators
    - _Source: `.dc.html` ANALYZING section_

  - [~] 10.4 Create result page
    - Convert the `<sc-if value="{{ isResult }}">` section into `app/templates/result.html` extending base
    - Preserve: two-column grid layout with verdict panel (family name, confidence meter, malicious/benign badge) and heatmap panel (canvas with entropy gradient legend, mean entropy, high-entropy %, structure label)
    - Include MITRE ATT&CK techniques list (technique ID, name, tactic) from RAG passages
    - Include explanation block with "Generated locally · grounded in retrieved MITRE ATT&CK context" footer
    - Include SHA-256 display and metadata grid (size, type, analyzed time, "Binary: Discarded")
    - Replace `{{ r.* }}` template vars with Jinja equivalents from `ScanRecord` / `ClassificationResult`
    - _Source: `.dc.html` RESULT section_

  - [~] 10.5 Create error page and history page
    - Convert the `<sc-if value="{{ isError }}">` section into `app/templates/error.html` — show rejection reason, file name, first-bytes vs expected MZ, "Try another file" button
    - Convert the `<sc-if value="{{ isHistory }}">` section into `app/templates/history.html` — grid table with sample hash, verdict (color dot + label), confidence, timestamp; empty state with "No scans yet" + CTA
    - Ensure all templates are accessible (semantic HTML, ARIA labels, keyboard navigation, role="alert" on error)
    - _Source: `.dc.html` ERROR and HISTORY sections_

- [~] 11. Checkpoint — Ensure backend and UI integration works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Docker Compose deployment
  - [~] 12.1 Create Dockerfile for application container
    - Create `Dockerfile` with Python 3.11 slim base
    - Install system dependencies for PyTorch CPU, Pillow
    - Copy requirements.txt, install Python deps
    - Copy app/, data/knowledge_base/, and models/ directories
    - Create data/ directories (chromadb, heatmaps)
    - Expose port 8000, CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
    - _Requirements: Docker Compose deployment from design architecture_

  - [~] 12.2 Create Docker Compose configuration
    - Create `docker-compose.yml` with two services:
      - `app`: build from Dockerfile, port 8000:8000, volume mounts for data/ and models/, depends_on ollama, environment variables for ENTROSIGHT_ settings, healthcheck
      - `ollama`: ollama/ollama:latest image, port 11434, volume for model storage, healthcheck
    - Add startup script or entrypoint to pull Ollama model (mistral) on first run
    - Create `.env.example` with all configurable environment variables
    - _Requirements: Architecture diagram, Ollama container service_

- [ ] 13. End-to-end integration and wiring
  - [~] 13.1 Wire all components together and verify startup
    - Ensure lifespan correctly initializes all components in order
    - Verify model checkpoint loading with a dummy/placeholder .pth file (or document the path expectation for teammate's checkpoint)
    - Verify ChromaDB knowledge base ingestion on first startup
    - Verify Ollama connectivity check on startup (log warning if unavailable)
    - Test full request lifecycle: upload → validate → heatmap → classify → RAG → explain → store → display
    - _Requirements: Algorithm 5, full sequence diagram, Error Scenarios 2-6_

  - [ ]* 13.2 Write integration tests for scan pipeline
    - Test complete scan with a small synthetic PE file (MZ header + random bytes)
    - Test deduplication: scanning same file twice returns consistent results
    - Test error paths: invalid file returns 422, missing model returns startup error
    - Test graceful degradation: scan completes with fallback explanation when Ollama is down
    - _Requirements: Integration Testing Strategy, Properties 1, 6, 7, 9_

- [~] 14. Final checkpoint — Full system verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements and design properties for traceability
- Checkpoints ensure incremental validation at natural break points
- The `.pth` checkpoint file comes from a teammate — the Dev builds only the loading/inference wrapper
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases using pytest
- All code is Python targeting Docker deployment with CPU-only PyTorch inference

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1", "6.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "3.2", "3.3", "4.1", "6.2"] },
    { "id": 4, "tasks": ["4.2", "7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "8.2"] },
    { "id": 6, "tasks": ["9.1"] },
    { "id": 7, "tasks": ["9.2"] },
    { "id": 8, "tasks": ["10.1", "10.2"] },
    { "id": 9, "tasks": ["10.3", "10.4", "10.5"] },
    { "id": 10, "tasks": ["12.1"] },
    { "id": 11, "tasks": ["12.2"] },
    { "id": 12, "tasks": ["13.1"] },
    { "id": 13, "tasks": ["13.2"] }
  ]
}
```
