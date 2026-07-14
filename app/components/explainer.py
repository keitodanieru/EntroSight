"""ExplanationGenerator — Ollama LLM-powered threat intelligence explanations."""

from __future__ import annotations

import time

import httpx

from app.models import ExplanationResult, RetrievedPassage


class ExplanationGenerator:
    """Generates plain-language threat explanations using Ollama LLM with RAG context."""

    FALLBACK_MESSAGE = "Explanation unavailable - LLM service temporarily unreachable"

    UNKNOWN_MESSAGE = (
        "The classifier could not determine this file's family with sufficient "
        "confidence. This does not necessarily indicate the file is safe or malicious — "
        "it may be a file type the model was not trained on, or an uncommon variant. "
        "Consider submitting to a multi-engine scanner for a second opinion."
    )

    SYSTEM_PROMPT = (
        "You are a malware analyst providing clear, concise threat intelligence "
        "explanations. Explain what the detected malware family does, its typical "
        "behaviors, and potential impact. Use plain language suitable for a "
        "security analyst. Keep response under 200 words."
    )

    def __init__(
        self,
        ollama_base_url: str,
        model_name: str = "mistral",
        max_tokens: int = 512,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialize Ollama client connection.

        Args:
            ollama_base_url: Base URL for the Ollama API (e.g., "http://ollama:11434").
            model_name: LLM model to use for generation.
            max_tokens: Maximum tokens to generate (num_predict).
            timeout_seconds: Timeout for the Ollama API call.
        """
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        label: str,
        confidence: float,
        context_passages: list[RetrievedPassage],
    ) -> ExplanationResult:
        """Generate plain-language explanation of classification.

        Formats a system prompt (malware analyst role) and a user prompt
        with the classification result and RAG context passages, then calls
        Ollama's /api/generate endpoint.

        Args:
            label: Predicted malware family label.
            confidence: Classification confidence (0.0 to 1.0).
            context_passages: RAG-retrieved MITRE ATT&CK passages (may be empty).

        Returns:
            ExplanationResult with explanation text, generation time, and model name.
        """
        start_time = time.perf_counter()

        # Short-circuit for Unknown — no point calling the LLM
        if label == "Unknown":
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExplanationResult(
                explanation_text=self.UNKNOWN_MESSAGE,
                generation_time_ms=elapsed_ms,
                model_used="none",
            )

        # Build context from RAG passages
        if context_passages:
            context_text = "\n\n".join(
                f"[{p.technique_id}] {p.technique_name}: {p.text}"
                for p in context_passages
            )
            user_prompt = (
                f"Classification Result:\n"
                f"- Family: {label}\n"
                f"- Confidence: {confidence:.1%}\n\n"
                f"Relevant MITRE ATT&CK Techniques:\n{context_text}\n\n"
                f"Provide a plain-language explanation of this malware family, "
                f"its typical behaviors, and recommended actions."
            )
        else:
            # Handle empty context_passages gracefully
            user_prompt = (
                f"Classification Result:\n"
                f"- Family: {label}\n"
                f"- Confidence: {confidence:.1%}\n\n"
                f"No MITRE ATT&CK context available.\n\n"
                f"Provide a plain-language explanation of this malware family, "
                f"its typical behaviors, and recommended actions."
            )

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds)
            ) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "system": self.SYSTEM_PROMPT,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {"num_predict": self.max_tokens},
                    },
                )
                response.raise_for_status()
                result_data = response.json()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ExplanationResult(
                explanation_text=result_data["response"].strip(),
                generation_time_ms=elapsed_ms,
                model_used=self.model_name,
            )

        except (httpx.TimeoutException, httpx.ConnectError):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExplanationResult(
                explanation_text=self.FALLBACK_MESSAGE,
                generation_time_ms=elapsed_ms,
                model_used=self.model_name,
            )
