"""Local Ollama lifecycle management.

Provides helpers to detect a running Ollama server, optionally start one
(`ollama serve`) when running natively, and ensure the configured model is
pulled. This is a best-effort convenience for local/non-Docker runs: every
failure path degrades gracefully (logs a warning and returns), because the
explanation step already falls back cleanly when the LLM is unreachable.

In the Docker stack the dedicated `ollama` container serves the model, so the
app detects it as already reachable and never spawns a process.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


async def is_ollama_up(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if the Ollama HTTP API answers at ``base_url``."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError, OSError):
        return False


async def model_present(base_url: str, model: str, timeout: float = 5.0) -> bool:
    """Return True if ``model`` is already available in the Ollama instance.

    Ollama tags include an implicit ":latest" suffix, so a match on the base
    name (before any tag) counts as present.
    """
    url = base_url.rstrip("/") + "/api/tags"
    wanted = model.split(":", 1)[0]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return False
    for entry in data.get("models", []):
        name = str(entry.get("name", ""))
        if name == model or name.split(":", 1)[0] == wanted:
            return True
    return False


def _is_local_url(base_url: str) -> bool:
    """Return True if ``base_url`` points at the local machine."""
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


async def _wait_until_up(base_url: str, timeout_seconds: int) -> bool:
    """Poll the Ollama endpoint until it responds or the timeout elapses."""
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        if await is_ollama_up(base_url):
            return True
        await asyncio.sleep(0.5)
    return False


async def _pull_model(binary: str, model: str) -> None:
    """Pull ``model`` via the Ollama CLI as a background coroutine.

    The first pull can download several GB and take minutes; this runs without
    blocking startup. The explanation step falls back gracefully until the
    model becomes available.
    """
    logger.info("Pulling Ollama model '%s' (this may take a while on first run)...", model)
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "pull",
            model,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("Ollama model '%s' is ready.", model)
        else:
            logger.warning(
                "Failed to pull Ollama model '%s' (exit %s): %s",
                model,
                proc.returncode,
                (stderr or b"").decode(errors="replace").strip(),
            )
    except Exception:  # noqa: BLE001 - background best-effort task
        logger.exception("Error while pulling Ollama model '%s'", model)


def _spawn_ollama_serve(binary: str) -> subprocess.Popen | None:
    """Start `ollama serve` as a detached child process, or return None on failure."""
    creationflags = 0
    if sys.platform == "win32":
        # Avoid opening a console window for the background server.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to spawn 'ollama serve'")
        return None


async def ensure_ollama(settings) -> subprocess.Popen | None:
    """Ensure an Ollama server is running and the model is (being) pulled.

    Returns the spawned ``subprocess.Popen`` handle if this call started the
    server (so the caller can terminate it on shutdown), or None if Ollama was
    already running or could not be started.
    """
    base_url = settings.ollama_base_url
    model = settings.ollama_model

    # Already reachable (e.g. Docker ollama container, or a pre-running server).
    if await is_ollama_up(base_url):
        logger.info("Ollama is reachable at %s", base_url)
        if settings.ollama_auto_pull_model and not await model_present(base_url, model):
            binary = shutil.which("ollama")
            if binary:
                asyncio.create_task(_pull_model(binary, model))
            else:
                logger.info(
                    "Model '%s' not present and no local ollama CLI to pull it; "
                    "explanations will use the fallback until it is available.",
                    model,
                )
        return None

    if not settings.ollama_autostart:
        logger.warning(
            "Ollama not reachable at %s and autostart is disabled; "
            "explanations will use the fallback message.",
            base_url,
        )
        return None

    binary = shutil.which("ollama")
    if not binary:
        logger.warning(
            "Ollama autostart requested but the 'ollama' binary was not found on PATH. "
            "Install Ollama (https://ollama.com/download) to enable local explanations; "
            "the app will run with the fallback explanation message until then."
        )
        return None

    if not _is_local_url(base_url):
        logger.warning(
            "Ollama autostart will start a local server, but ENTROSIGHT_OLLAMA_BASE_URL "
            "is '%s' (not localhost). The app will not reach the local server. Set "
            "ENTROSIGHT_OLLAMA_BASE_URL=http://localhost:11434 for native runs.",
            base_url,
        )
        # Still attempt to start it; the operator may fix the URL and retry.

    logger.info("Starting local 'ollama serve'...")
    proc = _spawn_ollama_serve(binary)
    if proc is None:
        return None

    if await _wait_until_up(base_url, settings.ollama_startup_timeout_seconds):
        logger.info("Local Ollama server is up at %s", base_url)
        if settings.ollama_auto_pull_model and not await model_present(base_url, model):
            asyncio.create_task(_pull_model(binary, model))
    else:
        logger.warning(
            "Started 'ollama serve' but it did not become reachable at %s within %ss; "
            "explanations will use the fallback message.",
            base_url,
            settings.ollama_startup_timeout_seconds,
        )

    return proc


def stop_ollama(proc: subprocess.Popen | None) -> None:
    """Terminate a previously spawned Ollama server process, if any."""
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping local Ollama server (pid %s)...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
