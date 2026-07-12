"""Training configuration — all hyperparameters and paths in one place."""

from pathlib import Path


# === Paths ===
# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where raw PE binaries are extracted (temporary, delete after dataset generation)
RAW_SAMPLES_DIR = PROJECT_ROOT / "raw_samples"

# Where generated heatmap tensors are saved
DATASET_DIR = PROJECT_ROOT / "dataset"

# Where training outputs (checkpoints, logs, plots) are saved
OUTPUT_DIR = PROJECT_ROOT / "training_outputs"

# Final model destination for the app
MODELS_DIR = PROJECT_ROOT / "models"


# === Dataset Generation ===
# Maximum samples per class (for balanced dataset)
MAX_SAMPLES_PER_CLASS = 1500

# Train/validation/test split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Random seed for reproducibility
SEED = 42


# === Class Labels (must match app/components/classifier.py) ===
CLASS_LABELS = [
    "AgentTesla",
    "Remcos",
    "DCRat",
    "FormBook",
    "RedLine",
    "AsyncRAT",
    "Benign",
]


# === Model Architecture ===
# Freeze layers up to and including this layer
# Options: "layer1", "layer2", "layer3", or None (train everything)
FREEZE_THROUGH = "layer2"

# Number of output classes
NUM_CLASSES = len(CLASS_LABELS)


# === Training Hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
NUM_WORKERS = 4  # DataLoader workers (adjust based on CPU cores)

# Mixed precision training (uses GPU Tensor Cores for ~2x speedup)
USE_AMP = True


# === Early Stopping ===
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 0.001


# === Learning Rate Scheduler ===
# CosineAnnealingWarmRestarts parameters
COSINE_T_0 = 10  # Restart period (epochs)
COSINE_T_MULT = 2  # Period multiplier after each restart


# === Data Augmentation (training only) ===
# NOTE: Only spatial augmentations are used. ColorJitter is intentionally excluded
# because the heatmap encodes entropy as pixel intensity — perturbing brightness/contrast
# would alter the actual entropy signal, injecting label noise rather than meaningful variation.
AUGMENTATION = {
    "random_horizontal_flip": 0.5,
    "random_rotation_degrees": 10,
}


# === ImageNet Normalization (must match inference in classifier.py) ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
