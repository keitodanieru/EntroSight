"""Tests for ExplanationGenerator component."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.components.explainer import ExplanationGenerator
from app.models import ExplanationResult, RetrievedPassage


@pytest.fixture
def generator() -> ExplanationGenerator:
    """Create an ExplanationGenerator with test configuration."""
    return ExplanationGenerator(
        ollama_base_url="http://localhost:11434",
        model_name="mistral",
        max_tokens=512,
        timeout_seconds=30,
    )


@pytest.fixture
def sample_passages() -> list[RetrievedPassage]:
    """Sample RAG context passages for testing."""
    return [
        RetrievedPassage(
            text="Process injection allows running code in another process's address space.",
            technique_id="T1055",
            technique_name="Process Injection",
            relevance_score=0.85,
        ),
        RetrievedPassage(
            text="Keylogging captures keystrokes to steal credentials.",
            technique_id="T1056.001",
            technique_name="Input Capture: Keylogging",
            relevance_score=0.72,
        ),
    ]


def _mock_successful_response(text: str = "This is a test explanation.") -> MagicMock:
    """Create a mock httpx response with successful JSON body."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": text}
    return mock_response


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestExplanationGeneratorSuccess:
    """Successful generation returns ExplanationResult with correct data."""

    @pytest.mark.asyncio
    async def test_successful_generation_returns_explanation_text(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Successful generation returns ExplanationResult with explanation text from mocked response."""
        expected_text = "AgentTesla is a keylogger and info-stealer."
        mock_response = _mock_successful_response(f"  {expected_text}  ")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="AgentTesla",
                confidence=0.92,
                context_passages=sample_passages,
            )

        assert isinstance(result, ExplanationResult)
        assert result.explanation_text == expected_text

    @pytest.mark.asyncio
    async def test_model_used_matches_configured_model(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """model_used matches the configured model name."""
        mock_response = _mock_successful_response()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="Remcos",
                confidence=0.85,
                context_passages=sample_passages,
            )

        assert result.model_used == "mistral"

    @pytest.mark.asyncio
    async def test_generation_time_ms_is_positive_on_success(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """generation_time_ms is positive on successful generation."""
        mock_response = _mock_successful_response()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="DCRat",
                confidence=0.78,
                context_passages=sample_passages,
            )

        assert result.generation_time_ms > 0


class TestExplanationGeneratorTimeout:
    """Timeout handling returns fallback message."""

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback_message(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Timeout returns fallback message."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="AsyncRAT",
                confidence=0.88,
                context_passages=sample_passages,
            )

        assert result.explanation_text == ExplanationGenerator.FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_timeout_generation_time_ms_is_positive(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """generation_time_ms is positive even on timeout."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="Formbook",
                confidence=0.65,
                context_passages=sample_passages,
            )

        assert result.generation_time_ms > 0

    @pytest.mark.asyncio
    async def test_timeout_model_used_still_set(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """model_used is still correctly set on timeout."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="Formbook",
                confidence=0.65,
                context_passages=[],
            )

        assert result.model_used == "mistral"


class TestExplanationGeneratorConnectionError:
    """Connection error returns fallback message."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_fallback_message(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Connection error returns fallback message."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="RedLineStealer",
                confidence=0.91,
                context_passages=sample_passages,
            )

        assert result.explanation_text == ExplanationGenerator.FALLBACK_MESSAGE

    @pytest.mark.asyncio
    async def test_connection_error_generation_time_ms_is_positive(
        self, generator: ExplanationGenerator
    ) -> None:
        """generation_time_ms is positive even on connection error."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="Remcos",
                confidence=0.73,
                context_passages=[],
            )

        assert result.generation_time_ms > 0


class TestExplanationGeneratorPromptFormatting:
    """Prompt formatting with empty and non-empty context passages."""

    @pytest.mark.asyncio
    async def test_empty_context_passages_handled(
        self, generator: ExplanationGenerator
    ) -> None:
        """Empty context_passages are handled (prompt still works)."""
        mock_response = _mock_successful_response("Explanation without context.")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="Benign",
                confidence=0.99,
                context_passages=[],
            )

        assert result.explanation_text == "Explanation without context."
        # Verify the post was called with a prompt mentioning "No MITRE ATT&CK context"
        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "No MITRE ATT&CK context available" in json_body["prompt"]

    @pytest.mark.asyncio
    async def test_non_empty_context_passages_formatted_in_prompt(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Non-empty context passages are formatted in the prompt."""
        mock_response = _mock_successful_response("Detailed explanation with context.")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(
                label="AgentTesla",
                confidence=0.92,
                context_passages=sample_passages,
            )

        assert result.explanation_text == "Detailed explanation with context."
        # Verify the post was called with passages formatted in the prompt
        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt = json_body["prompt"]
        assert "[T1055] Process Injection:" in prompt
        assert "[T1056.001] Input Capture: Keylogging:" in prompt
        assert "Relevant MITRE ATT&CK Techniques:" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_label_and_confidence(
        self, generator: ExplanationGenerator, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Prompt includes the classification label and confidence percentage."""
        mock_response = _mock_successful_response("Some explanation.")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await generator.generate(
                label="DCRat",
                confidence=0.756,
                context_passages=sample_passages,
            )

        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt = json_body["prompt"]
        assert "DCRat" in prompt
        assert "75.6%" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_is_malware_analyst_role(
        self, generator: ExplanationGenerator
    ) -> None:
        """System prompt uses the malware analyst role prompt."""
        mock_response = _mock_successful_response("Test.")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await generator.generate(
                label="Benign",
                confidence=0.95,
                context_passages=[],
            )

        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "malware analyst" in json_body["system"]
