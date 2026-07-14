"""FileValidator — PE file validation component."""

import hashlib

from app.models import ValidationResult


class FileValidator:
    """Validates uploaded files as legitimate PE binaries.

    Checks performed:
    1. File extension against allowlist (.exe, .dll, .sys)
    2. File size against maximum limit (50 MB)
    3. MZ signature verification (first 2 bytes)
    4. SHA-256 hash computation for deduplication
    """

    ALLOWED_EXTENSIONS: set[str] = {".exe", ".dll", ".sys"}
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

    def validate(self, filename: str, file_bytes: bytes) -> ValidationResult:
        """Validate an uploaded file as a legitimate PE binary.

        Args:
            filename: Original filename including extension.
            file_bytes: Raw file content as bytes.

        Returns:
            ValidationResult with is_valid=True and file_hash on success,
            or is_valid=False with error_message on failure.
        """
        # Check size
        if len(file_bytes) > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE // (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                error_message=f"File too large. Maximum size: {max_mb} MB",
            )

        # Check MZ signature
        if len(file_bytes) < 2 or file_bytes[:2] != b"MZ":
            return ValidationResult(
                is_valid=False,
                error_message="Invalid PE file: missing MZ signature",
            )

        # Compute hash
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        return ValidationResult(is_valid=True, file_hash=sha256)
