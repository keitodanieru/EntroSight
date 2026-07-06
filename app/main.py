"""FastAPI application: routes, lifespan, and component wiring for EntroSight."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.components.classifier import MalwareClassifier
from app.components.database import ScanHistoryDB
from app.components.explainer import ExplanationGenerator
from app.components.heatmap import EntropyHeatmapGenerator
from app.components.rag import RAGEngine
from app.components.validator import FileValidator
from app.config import AppSettings
from app.knowledge_base_loader import load_knowledge_base
from app.models import ScanResponse, ScanStatusResponse
from app.ollama_manager import ensure_ollama, stop_ollama
from app.scan import (
    AppComponents,
    execute_scan,
    get_scan_status,
    register_scan,
)

logger = logging.getLogger(__name__)

STATIC_DIR = "app/static"
TEMPLATES_DIR = "app/templates"

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Settings are read at import time (in addition to lifespan) so the heatmap
# storage directory can be mounted as a static route before the app starts
# serving requests.
_module_settings = AppSettings()
os.makedirs(_module_settings.heatmap_storage_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize all pipeline components on startup, clean up on shutdown.

    Loads the model checkpoint, opens the scan history database, connects to
    ChromaDB, and ingests the MITRE ATT&CK knowledge base if the collection
    is empty. All components are attached to ``app.state`` for use by routes
    and the background scan task.
    """
    settings = AppSettings()

    validator = FileValidator()
    heatmap_generator = EntropyHeatmapGenerator()
    classifier = MalwareClassifier(
        checkpoint_path=settings.model_checkpoint_path, device="cpu"
    )
    rag_engine = RAGEngine(
        collection_name=settings.rag_collection_name,
        persist_directory=settings.chromadb_path,
    )
    explanation_generator = ExplanationGenerator(
        ollama_base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        max_tokens=settings.explanation_max_tokens,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    db = ScanHistoryDB(db_path=settings.database_path)
    await db.initialize()

    # Best-effort: ensure a local Ollama server is running (native runs) and
    # the model is pulled. No-op when Ollama is already reachable (Docker) or
    # not installed — the explanation step falls back gracefully either way.
    ollama_proc = await ensure_ollama(settings)

    # Ingest the MITRE ATT&CK knowledge base if the collection is empty.
    try:
        if rag_engine.collection.count() == 0:
            load_knowledge_base(rag_engine)
    except Exception:
        logger.exception("Failed to ingest MITRE ATT&CK knowledge base on startup")

    components = AppComponents(
        validator=validator,
        heatmap_generator=heatmap_generator,
        classifier=classifier,
        rag_engine=rag_engine,
        explanation_generator=explanation_generator,
        db=db,
    )

    app.state.settings = settings
    app.state.components = components

    try:
        yield
    finally:
        await db.close()
        # Only stops a server this process spawned; leaves external/Docker
        # Ollama instances untouched.
        stop_ollama(ollama_proc)


app = FastAPI(title="EntroSight", lifespan=lifespan)
# More specific mount must be registered before the general "/static" mount,
# since Starlette matches mounted routes in registration order.
app.mount(
    "/static/heatmaps",
    StaticFiles(directory=_module_settings.heatmap_storage_dir),
    name="heatmaps",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _is_htmx(request: Request) -> bool:
    """Return True if the request originates from HTMX (expects an HTML fragment)."""
    return request.headers.get("HX-Request") == "true"


def _heatmap_url_for(record) -> str:
    """Build the browser-facing heatmap URL served from the static mount."""
    return f"/static/heatmaps/{record.sha256}.png"


def _record_to_scan_response(record) -> ScanResponse:
    """Convert a ScanRecord into the API-facing ScanResponse shape."""
    return ScanResponse(
        scan_id=record.id,
        sha256=record.sha256,
        filename=record.filename,
        predicted_label=record.predicted_label,
        confidence=record.confidence,
        explanation=record.explanation,
        heatmap_url=f"/static/heatmaps/{record.sha256}.png",
        timestamp=record.timestamp,
        total_time_ms=record.total_time_ms,
    )


def _status_entry_to_response(entry: dict) -> ScanStatusResponse:
    """Convert an in-memory scan status entry into a ScanStatusResponse."""
    result = entry.get("result")
    return ScanStatusResponse(
        scan_id=entry["scan_id"],
        status=entry["status"],
        progress_stage=entry.get("progress_stage"),
        result=_record_to_scan_response(result) if result is not None else None,
        error_message=entry.get("error_message"),
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.post("/api/scan", status_code=202)
async def submit_scan(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    """Accept an uploaded PE file and launch the scan pipeline in the background.

    Returns immediately with a scan_id so the client can poll
    ``GET /api/scan/{scan_id}/status`` for progress and results.
    """
    file_bytes = await file.read()
    scan_id = str(uuid4())

    register_scan(scan_id, filename=file.filename)

    background_tasks.add_task(
        execute_scan,
        scan_id=scan_id,
        file_bytes=file_bytes,
        filename=file.filename,
        settings=request.app.state.settings,
        components=request.app.state.components,
    )

    # HTMX clients expect an HTML fragment to swap into #result-container:
    # render the polling partial so the UI can track progress. Non-HTMX
    # (programmatic/API) clients get the JSON 202 contract.
    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/scan_status.html",
            {
                "scan_id": scan_id,
                "status": "pending",
                "progress_stage": None,
                "filename": file.filename,
                "error_message": None,
            },
            status_code=202,
        )

    return JSONResponse(
        status_code=202,
        content={"scan_id": scan_id, "status": "pending"},
    )


@app.get("/api/scan/{scan_id}/status")
async def scan_status(request: Request, scan_id: str):
    """Return the current status of a scan.

    HTMX clients receive an HTML fragment (the polling partial). When the scan
    completes, the response carries an ``HX-Redirect`` header so the browser
    navigates to the full result page. Non-HTMX clients receive JSON.
    """
    entry = get_scan_status(scan_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    if _is_htmx(request):
        result = entry.get("result")
        # On completion, send the browser to the rendered result page.
        if entry["status"] == "complete" and result is not None:
            return Response(status_code=200, headers={"HX-Redirect": f"/scan/{result.id}"})
        return templates.TemplateResponse(
            request,
            "partials/scan_status.html",
            {
                "scan_id": entry["scan_id"],
                "status": entry["status"],
                "progress_stage": entry.get("progress_stage"),
                "filename": entry.get("filename"),
                "error_message": entry.get("error_message"),
            },
        )

    return _status_entry_to_response(entry)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse(
        request, "index.html", {"isUpload": True}
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Render the scan history page."""
    db: ScanHistoryDB = request.app.state.components.db
    recent_scans = await db.list_recent(limit=20)
    return templates.TemplateResponse(
        request, "history.html", {"isHistory": True, "scans": recent_scans}
    )


@app.get("/scan/{scan_id}", response_class=HTMLResponse)
async def scan_result_page(request: Request, scan_id: int):
    """Render an individual scan result page."""
    db: ScanHistoryDB = request.app.state.components.db
    record = await db.get_result(scan_id)
    if record is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"isError": True, "error_message": "Scan not found"},
            status_code=404,
        )
    # Attach a browser-facing heatmap URL (the stored heatmap_path is a
    # filesystem path and is not directly servable).
    record.heatmap_url = _heatmap_url_for(record)
    return templates.TemplateResponse(
        request, "result.html", {"isResult": True, "r": record}
    )
