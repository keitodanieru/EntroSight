"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application configuration with ENTROSIGHT_ env prefix."""

    # Model settings
    model_checkpoint_path: str = "models/resnet50_malware.pth"
    class_labels: list[str] = [
        "AgentTesla",
        "Remcos",
        "DCRat",
        "AsyncRAT",
        "RedLineStealer",
        "Formbook",
        "Benign",
    ]

    # Processing settings
    max_file_size_mb: int = 50
    entropy_block_size: int = 256
    heatmap_image_size: int = 256

    # RAG settings
    chromadb_path: str = "data/chromadb"
    rag_collection_name: str = "mitre_attack"
    rag_top_k: int = 5

    # Ollama settings
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "mistral"
    explanation_max_tokens: int = 512
    ollama_timeout_seconds: int = 60
    # When true, the app will try to start a local `ollama serve` process on
    # startup if the configured Ollama endpoint is not already reachable, and
    # pull the configured model. Intended for native (non-Docker) runs; in the
    # Docker stack the separate ollama container serves this role, so the app
    # simply detects it as already-up and skips spawning.
    ollama_autostart: bool = True
    # Seconds to wait for a spawned `ollama serve` to become reachable.
    ollama_startup_timeout_seconds: int = 30
    # When true, pull the configured model on startup if it isn't present.
    ollama_auto_pull_model: bool = True

    # Database settings
    database_path: str = "data/scans.db"

    # Storage settings
    heatmap_storage_dir: str = "data/heatmaps"

    class Config:
        env_prefix = "ENTROSIGHT_"
