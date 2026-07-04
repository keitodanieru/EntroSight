"""Tests for FileValidator component."""

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.components.validator import FileValidator
from app.models import ValidationResult


@pytest.fixture
def validator() -> FileValidator:
    return FileValidator()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestFileValidatorExtension:
    """Extension allowlist checks."""

    def test_accepts_exe(self, validator: FileValidator) -> None:
        result = validator.validate("malware.exe", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_accepts_dll(self, validator: FileValidator) -> None:
        result = validator.validate("library.dll", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_accepts_sys(self, validator: FileValidator) -> None:
        result = validator.validate("driver.sys", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_rejects_txt(self, validator: FileValidator) -> None:
        result = validator.validate("readme.txt", b"MZ" + b"\x00" * 100)
        assert result.is_valid is False
        assert "Invalid file type" in result.error_message

    def test_rejects_py(self, validator: FileValidator) -> None:
        result = validator.validate("script.py", b"MZ" + b"\x00" * 100)
        assert result.is_valid is False

    def test_case_insensitive_extension(self, validator: FileValidator) -> None:
        result = validator.validate("MALWARE.EXE", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_rejects_no_extension(self, validator: FileValidator) -> None:
        result = validator.validate("noext", b"MZ" + b"\x00" * 100)
        assert result.is_valid is False


class TestFileValidatorSize:
    """File size enforcement checks."""

    def test_accepts_small_file(self, validator: FileValidator) -> None:
        result = validator.validate("small.exe", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_accepts_file_at_limit(self, validator: FileValidator) -> None:
        data = b"MZ" + b"\x00" * (50 * 1024 * 1024 - 2)
        result = validator.validate("exact.exe", data)
        assert result.is_valid is True

    def test_rejects_file_over_limit(self, validator: FileValidator) -> None:
        data = b"MZ" + b"\x00" * (50 * 1024 * 1024)
        result = validator.validate("big.exe", data)
        assert result.is_valid is False
        assert "File too large" in result.error_message


class TestFileValidatorSignature:
    """MZ signature verification checks."""

    def test_accepts_valid_mz(self, validator: FileValidator) -> None:
        result = validator.validate("test.exe", b"MZ" + b"\x00" * 100)
        assert result.is_valid is True

    def test_rejects_missing_mz(self, validator: FileValidator) -> None:
        result = validator.validate("bad.exe", b"PK" + b"\x00" * 100)
        assert result.is_valid is False
        assert "missing MZ signature" in result.error_message

    def test_rejects_empty_file(self, validator: FileValidator) -> None:
        result = validator.validate("empty.exe", b"")
        assert result.is_valid is False
        assert "missing MZ signature" in result.error_message

    def test_rejects_one_byte(self, validator: FileValidator) -> None:
        result = validator.validate("tiny.exe", b"M")
        assert result.is_valid is False
        assert "missing MZ signature" in result.error_message


class TestFileValidatorHash:
    """SHA-256 hash computation checks."""

    def test_returns_correct_hash(self, validator: FileValidator) -> None:
        data = b"MZ" + b"\x00" * 100
        expected = hashlib.sha256(data).hexdigest()
        result = validator.validate("test.exe", data)
        assert result.is_valid is True
        assert result.file_hash == expected

    def test_hash_is_none_on_failure(self, validator: FileValidator) -> None:
        result = validator.validate("bad.txt", b"MZ" + b"\x00" * 100)
        assert result.is_valid is False
        assert result.file_hash is None


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestValidationCompleteness:
    """**Validates: Requirements 1.2** — Property 2 (Validation Completeness).

    For all inputs, either validation passes (is_valid=True with file_hash)
    or it fails (is_valid=False with error_message, no file_hash).
    """

    @given(
        filename=st.text(min_size=1, max_size=50),
        file_bytes=st.binary(min_size=0, max_size=1024),
    )
    @settings(max_examples=200)
    def test_validation_always_returns_complete_result(
        self, filename: str, file_bytes: bytes
    ) -> None:
        validator = FileValidator()
        result = validator.validate(filename, file_bytes)

        assert isinstance(result, ValidationResult)
        if result.is_valid:
            # Success: must have hash, no error
            assert result.file_hash is not None
            assert len(result.file_hash) == 64  # SHA-256 hex length
            assert result.error_message is None
        else:
            # Failure: must have error, no hash
            assert result.error_message is not None
            assert len(result.error_message) > 0
            assert result.file_hash is None


class TestExtensionSignatureAlignment:
    """**Validates: Requirements 1.2** — Property 10 (Extension-Signature Alignment).

    For all files that pass validation, they have both a valid extension
    AND a valid MZ signature.
    """

    @given(
        filename=st.text(min_size=1, max_size=50),
        file_bytes=st.binary(min_size=0, max_size=1024),
    )
    @settings(max_examples=200)
    def test_valid_files_have_extension_and_signature(
        self, filename: str, file_bytes: bytes
    ) -> None:
        from pathlib import Path

        validator = FileValidator()
        result = validator.validate(filename, file_bytes)

        if result.is_valid:
            # Must have valid extension
            ext = Path(filename).suffix.lower()
            assert ext in validator.ALLOWED_EXTENSIONS

            # Must have MZ signature
            assert len(file_bytes) >= 2
            assert file_bytes[:2] == b"MZ"
