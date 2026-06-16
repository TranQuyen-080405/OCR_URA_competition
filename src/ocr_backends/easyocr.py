"""EasyOCR backend (legacy fallback)."""

from __future__ import annotations

import numpy as np
import easyocr

from src.ocr_backends.types import OcrDetection, sort_detections


class EasyOcrBackend:
    name = "easyocr"

    def __init__(self, lang: str = "vi", use_gpu: bool = False) -> None:
        # EasyOCR: vi + en for mixed captions
        langs = ["vi", "en"] if lang == "vi" else [lang]
        self.lang = lang
        self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)

    def describe(self) -> str:
        return f"EasyOCR langs=vi+en ({'GPU' if self.reader.gpu else 'CPU'})"

    def read_detections(self, img_np: np.ndarray) -> list[OcrDetection]:
        try:
            results = self.reader.readtext(img_np, detail=1, paragraph=False)
        except Exception:
            return []
        detections = [(r[0], str(r[1]), float(r[2])) for r in results]
        return sort_detections(detections)
