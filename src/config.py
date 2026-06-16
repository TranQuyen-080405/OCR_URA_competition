"""Paths and constants (notebook Cell 2)."""

from pathlib import Path

IMAGE_DIR_CANDIDATES = (
    "test_images/images",
    "test_images",
    "images",
    "test/images",
    "test/test_images/images",
    "test/test_images",
)

IMAGE_ZIP_NAMES = (
    "test_images.zip",
    "images.zip",
    "test/test_images.zip",
    "test/images.zip",
)

LOCAL_DATASET_ROOTS = (
    Path("the-2nd-ura-hackathon"),
    Path("public_dataset"),
    Path("smce_dataset/test"),
    Path("."),
)

SAVE_EVERY = 50
OCR_CONF_THRESHOLD = 0.35
OCR_USE_GPU = False
OCR_BACKEND = "paddleocr"  # "paddleocr" | "easyocr"
OCR_LANG = "vi"              # PaddleOCR: vi = Vietnamese (latin model)

# ---------------------------------------------------------------------------
# Image preprocess before OCR (src/preprocess.py + baseline in ocr_engine)
# ---------------------------------------------------------------------------
# ENABLE_PREPROCESS=False  → ảnh gốc, không xử lý
# PREPROCESS_MODE="baseline" → resize + contrast + sharpen (~0.5 LB)
# PREPROCESS_MODE="advanced" → crop / CLAHE / deskew … (src/preprocess.py)
ENABLE_PREPROCESS = False
PREPROCESS_MODE = "baseline"  # "baseline" | "advanced"
USE_MULTIPASS_PREPROCESS = False  # chỉ áp dụng khi PREPROCESS_MODE == "advanced"

PREPROCESS_MAX_DIM = 1280
PREPROCESS_MIN_SIDE_FOR_UPSCALE = 800
PREPROCESS_UPSCALE_FACTOR = 1.75
PREPROCESS_UI_TOP_RATIO = 0.10
PREPROCESS_UI_BOTTOM_RATIO = 0.12
PREPROCESS_CENTER_WIDTH_RATIO = 0.92
PREPROCESS_CENTER_HEIGHT_RATIO = 0.78

SOLUTION_PATHS = (
    Path("smce_dataset/solution.csv"),
    Path("/kaggle/input/smce-solution/solution.csv"),
)

TRAIN_LABELS_CANDIDATES = (
    "train_labels.csv",
    "the-2nd-ura-hackathon/train_labels.csv",
    "public_dataset/train_labels.csv",
)


def describe_preprocess_config() -> str:
    if not ENABLE_PREPROCESS:
        return "preprocess OFF (raw image)"
    if PREPROCESS_MODE == "advanced":
        if USE_MULTIPASS_PREPROCESS:
            return "preprocess advanced + multi-pass (4 variants)"
        return "preprocess advanced single-pass"
    return "preprocess baseline (resize + contrast + sharpen)"
