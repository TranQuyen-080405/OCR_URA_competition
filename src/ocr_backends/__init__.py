"""OCR backend adapters (PaddleOCR, EasyOCR)."""

from __future__ import annotations

from src.ocr_backends.paddle import PaddleOcrBackend


def create_ocr_backend(backend: str, *, lang: str, use_gpu: bool):
    if backend == "paddleocr":
        return PaddleOcrBackend(lang=lang, use_gpu=use_gpu)
    if backend == "easyocr":
        from src.ocr_backends.easyocr import EasyOcrBackend

        return EasyOcrBackend(lang=lang, use_gpu=use_gpu)
    raise ValueError(f"Unknown OCR backend: {backend!r}. Use 'paddleocr' or 'easyocr'.")


__all__ = ["PaddleOcrBackend", "create_ocr_backend"]
