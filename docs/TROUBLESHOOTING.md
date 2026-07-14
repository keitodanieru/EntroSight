# Troubleshooting

This section covers common issues encountered when using EntroSight and how to resolve them.

---

## File Rejected / Invalid PE File

EntroSight validates every upload before processing begins. If validation fails, the file is rejected immediately and no further analysis occurs.

### "Invalid file type" Error

The uploaded file has an extension that is not in the allowlist. EntroSight only accepts `.exe`, `.dll`, and `.sys` files. Rename the file or confirm it is genuinely a Windows PE binary before re-uploading.

### "Invalid PE file: missing MZ signature" Error

The file has an accepted extension but does not begin with the two-byte `MZ` (hex `4D 5A`) header that all valid Windows PE files start with. This usually means:

- The file is corrupted or was truncated during transfer.
- The file was renamed from a different format (e.g., a `.zip` renamed to `.exe`).
- The file is empty or too short (less than 2 bytes).

Confirm the file is a genuine Windows executable by inspecting its first bytes with a hex editor or by verifying its source.

### "File too large" Error

Files exceeding the configured maximum size (50 MB by default) are rejected. If larger samples need to be supported, increase the `ENTROSIGHT_MAX_FILE_SIZE_MB` environment variable, keeping available system memory in mind since the entire file is loaded into RAM for entropy computation.

---

## Scan Fails or Times Out

Once a file passes validation, the scan pipeline runs as a background task. If any stage fails, the scan is marked as errored and the UI displays a "Scan failed" message with an error description.

### "Memory error during entropy computation"

The file's size caused the system to exhaust available RAM during heatmap generation. This is most likely to happen with files approaching the 50 MB limit on systems with limited memory. Reduce `ENTROSIGHT_MAX_FILE_SIZE_MB` or increase available RAM.

### "Failed to store scan result"

The classification and explanation completed, but the final database write failed. Common causes:

- The disk where `data/scans.db` resides is full.
- The SQLite database file is locked by another process.
- The `data/` directory permissions do not allow writes.

Check available disk space and file permissions on the `ENTROSIGHT_DATABASE_PATH` location.

### Generic or Unexpected Error

If the error message does not match the patterns above, an unexpected failure occurred in the classifier or heatmap generator. Check the application logs (`docker-compose logs app`) for the full traceback. The error message shown to the user is the raw exception text, which may not be user-friendly but will indicate the failing component.

---

## Explanation Shows Fallback Message

If the result page displays **"Explanation unavailable - LLM service temporarily unreachable"** instead of a plain-language explanation, it means the Ollama LLM service could not be reached or did not respond in time. **The scan itself still completes successfully** — the classification, confidence score, heatmap, and MITRE ATT&CK techniques are all present. Only the generated explanation is missing.

### Ollama Container Not Healthy (Docker)

Verify both containers are running and healthy:

```bash
docker-compose ps
```

The `ollama` service should show a `healthy` status. If it is restarting or unhealthy, check its logs:

```bash
docker-compose logs ollama
```

### Model Not Yet Pulled (First Startup)

On first startup, the `ollama-init` service downloads the Mistral model (~4 GB). Until this download finishes, explanations will fall back. Monitor progress with:

```bash
docker-compose logs -f ollama-init
```

Once the init service exits successfully, restart a scan to get a full explanation.

### Timeout (Default: 60 Seconds)

If Ollama is running but the model is overloaded or the system has limited CPU/RAM, generation may exceed the 60-second timeout. Increase `ENTROSIGHT_OLLAMA_TIMEOUT_SECONDS` if this happens consistently on your hardware.

### Native Run: Ollama Not Installed

When running outside Docker, EntroSight attempts to start a local `ollama serve` process automatically (controlled by `ENTROSIGHT_OLLAMA_AUTOSTART=true`). If the `ollama` binary is not found on PATH, the app logs a warning and runs without explanations. Install Ollama from [ollama.com/download](https://ollama.com/download) to enable local explanations.

### Native Run: Base URL Misconfigured

For non-Docker runs, `ENTROSIGHT_OLLAMA_BASE_URL` must point to `http://localhost:11434` (not `http://ollama:11434`, which is the Docker-internal hostname). If this is misconfigured, the app may start a local server but fail to reach it. Update your `.env` file:

```
ENTROSIGHT_OLLAMA_BASE_URL=http://localhost:11434
```

---

## Application Won't Start

If the application fails to start entirely (container exits immediately or the `uvicorn` process crashes), check these causes.

### Model Checkpoint Missing or Invalid

The `MalwareClassifier` loads the ResNet50 checkpoint during application startup. If the file at `ENTROSIGHT_MODEL_CHECKPOINT_PATH` (default: `models/resnet50_malware.pth`) does not exist or is corrupted, the app will crash on startup with a `FileNotFoundError` or PyTorch deserialization error.

**Resolution:** Ensure the trained `.pth` file from the ML teammate is placed at the configured path. In Docker, the `models/` directory is mounted as a volume — confirm the file exists on the host at `./models/resnet50_malware.pth`.

### Port Already in Use

If port `8000` (app) or `11434` (Ollama) is already occupied by another process, Docker will fail to bind and the container will not start.

**Resolution:** Stop the conflicting process, or remap the ports in `docker-compose.yml`:

```yaml
ports:
  - "9000:8000"  # Map to a different host port
```

### Docker Compose Dependency Failures

The `app` service depends on both `ollama` (healthy) and `ollama-init` (completed successfully). If either dependency fails, the app container will not start.

**Resolution:** Run `docker-compose logs` to identify which service failed and address its specific error before restarting.

---

## First Startup Takes a Long Time

On the very first run, expect a delay of several minutes while:

1. The Docker images are built (if not pre-pulled).
2. The `ollama-init` service downloads the Mistral model (~4 GB depending on network speed).
3. The knowledge base is ingested into ChromaDB (fast, but logged).

This is normal. Subsequent startups skip the model download and are significantly faster. Monitor progress with:

```bash
docker-compose logs -f
```

---

## No MITRE ATT&CK Context in Explanation

If the explanation is generated but does not reference specific MITRE ATT&CK techniques, the RAG retrieval step returned empty results. This happens when:

- The ChromaDB knowledge base was not ingested on startup (check logs for "Failed to ingest MITRE ATT&CK knowledge base on startup").
- The `data/chromadb/` directory was deleted or corrupted after initial ingestion.
- The predicted family label has no matching documents in the knowledge base (unlikely with the default 35-entry set).

**Resolution:** Delete the `data/chromadb/` directory and restart the application. The knowledge base will be re-ingested automatically on the next startup when the collection is detected as empty.

---

## Duplicate File Scans

Submitting the same file multiple times is allowed and does not produce an error. Each submission creates a new scan record in the history. EntroSight computes the SHA-256 hash for identification but does not block re-analysis of previously seen files. This is by design — results may differ if the LLM generates a slightly different explanation.

---

## Summary of Key Environment Variables for Troubleshooting

| Variable | Default | Relevant Issue |
|----------|---------|----------------|
| `ENTROSIGHT_MODEL_CHECKPOINT_PATH` | `models/resnet50_malware.pth` | App won't start |
| `ENTROSIGHT_OLLAMA_BASE_URL` | `http://ollama:11434` | Explanation fallback |
| `ENTROSIGHT_OLLAMA_MODEL` | `mistral` | Model pull failures |
| `ENTROSIGHT_OLLAMA_TIMEOUT_SECONDS` | `60` | Explanation timeout |
| `ENTROSIGHT_OLLAMA_AUTOSTART` | `true` | Native run behavior |
| `ENTROSIGHT_MAX_FILE_SIZE_MB` | `50` | File too large rejections |
| `ENTROSIGHT_DATABASE_PATH` | `data/scans.db` | DB write failures |
| `ENTROSIGHT_CHROMADB_PATH` | `data/chromadb` | Missing MITRE context |
