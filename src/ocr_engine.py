"""OCR engine with pluggable backend and configurable preprocess."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import (
    ENABLE_PREPROCESS,
    OCR_BACKEND,
    OCR_CONF_THRESHOLD,
    OCR_LANG,
    OCR_USE_GPU,
    PREPROCESS_MAX_DIM,
    PREPROCESS_MODE,
    USE_MULTIPASS_PREPROCESS,
    describe_preprocess_config,
)
from src.data_loader import DataBundle
from src.brand_rules import extract_product
from src.ocr_backends import create_ocr_backend
from src.ocr_backends.types import detections_to_text
from src.run_logger import RunLogger


def load_image(images_dir: Path, image_id: str) -> Image.Image | None:
    path = images_dir / f"{image_id}.jpg"
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except OSError:
        return None


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL → numpy without holding extra PIL buffer."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def preprocess_baseline(img: Image.Image, max_dim: int = PREPROCESS_MAX_DIM) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img.filter(ImageFilter.SHARPEN)


def prepare_image_for_ocr(img: Image.Image) -> Image.Image:
    if not ENABLE_PREPROCESS:
        return img
    if PREPROCESS_MODE == "advanced":
        from src.preprocess import preprocess_single_pass

        return preprocess_single_pass(img)
    return preprocess_baseline(img)


def postprocess_ocr(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if not tokens:
        return ""
    deduped = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != deduped[-1].lower():
            deduped.append(tok)
    return " ".join(deduped)


class OcrEngine:
    def __init__(self, data: DataBundle, predict_product: Callable[[str], str]):
        self.data = data
        self.predict_product = predict_product
        self.backend = create_ocr_backend(OCR_BACKEND, lang=OCR_LANG, use_gpu=OCR_USE_GPU)
        self._use_multipass = (
            ENABLE_PREPROCESS
            and PREPROCESS_MODE == "advanced"
            and USE_MULTIPASS_PREPROCESS
        )

    def _ocr_numpy(self, img_np: np.ndarray) -> str:
        detections = self.backend.read_detections(img_np)
        return postprocess_ocr(detections_to_text(detections, OCR_CONF_THRESHOLD))

    def _ocr_multipass(self, img: Image.Image) -> str:
        from src.preprocess import generate_multipass_variants, pil_to_rgb_numpy, score_ocr_detections

        best_score = -1.0
        best_text = ""
        for variant in generate_multipass_variants(img):
            try:
                detections = self.backend.read_detections(pil_to_rgb_numpy(variant.image))
                score, raw_text = score_ocr_detections(detections, OCR_CONF_THRESHOLD)
                if score > best_score:
                    best_score = score
                    best_text = postprocess_ocr(raw_text)
            except Exception:
                continue
        return best_text

    def run_ocr(self, image_id: str) -> tuple[str, str]:
        img = load_image(self.data.images_dir, image_id)
        if img is None:
            return "", ""

        try:
            if self._use_multipass:
                ocr_text = self._ocr_multipass(img)
            else:
                prepared = prepare_image_for_ocr(img)
                ocr_text = self._ocr_numpy(pil_to_numpy_rgb(prepared))
                if prepared is not img:
                    prepared.close()
        except MemoryError:
            gc.collect()
            ocr_text = ""
        finally:
            img.close()

        return ocr_text, self.predict_product(ocr_text)


def build_ocr_engine(
    data: DataBundle,
    predict_product: Callable[[str], str],
    logger: RunLogger,
) -> OcrEngine:
    logger.section("Cell 4 — OCR Engine")
    engine = OcrEngine(data, predict_product)
    logger.log(f"{engine.backend.describe()}, {describe_preprocess_config()}")
    if OCR_USE_GPU and hasattr(engine.backend, "use_gpu_effective"):
        b = engine.backend
        logger.log(
            f"  GPU check   : requested={b.use_gpu_requested} | "
            f"cuda_available={b.cuda_available} | effective={b.use_gpu_effective}"
        )

    logger.log("\nSmoke test on first image...")
    first_id = data.test_df["image_id"].iloc[0]
    ocr, prod = engine.run_ocr(first_id)
    prod_rules = extract_product(ocr)
    logger.log(f"  image_id    : {first_id}")
    preview = ocr[:80] + ("..." if len(ocr) > 80 else "")
    logger.log(f"  ocr_text    : {preview}")
    logger.log(f"  product     : {prod or '(empty)'}")
    if prod != prod_rules:
        logger.log(f"  rules-only  : {prod_rules or '(empty)'}  (train model override)")

    return engine
