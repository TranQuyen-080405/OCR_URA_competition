"""PaddleOCR backend — Vietnamese via lang='vi' (latin PP-OCRv3 rec)."""

from __future__ import annotations

import os
import warnings

import numpy as np
import paddle
from paddleocr import PaddleOCR

from src.ocr_backends.types import OcrDetection, sort_detections


def _cuda_ready() -> bool:
    return bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)


class PaddleOcrBackend:
    name = "paddleocr"

    def __init__(self, lang: str = "vi", use_gpu: bool = False) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.lang = lang
        self.use_gpu_requested = use_gpu
        self.cuda_available = _cuda_ready()
        self.use_gpu_effective = use_gpu and self.cuda_available

        if use_gpu and not self.cuda_available:
            warnings.warn(
                "OCR_USE_GPU=True nhưng đang dùng paddlepaddle bản CPU "
                "(compiled_with_cuda=False). PaddleOCR sẽ chạy CPU. "
                "Cài GPU: pip uninstall paddlepaddle -y && "
                "pip install paddlepaddle-gpu==2.6.2 -i "
                "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
                stacklevel=2,
            )

        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=self.use_gpu_effective,
            show_log=False,
        )

    def describe(self) -> str:
        if self.use_gpu_effective:
            return f"PaddleOCR lang={self.lang} (GPU active)"
        if self.use_gpu_requested:
            return (
                f"PaddleOCR lang={self.lang} (CPU — config GPU=True "
                f"nhưng thiếu paddlepaddle-gpu / CUDA)"
            )
        return f"PaddleOCR lang={self.lang} (CPU)"

    def read_detections(self, img_np: np.ndarray) -> list[OcrDetection]:
        try:
            result = self.reader.ocr(img_np, cls=True)
        except Exception:
            return []
        if not result or not result[0]:
            return []
        detections: list[OcrDetection] = []
        for line in result[0]:
            box, (text, conf) = line
            detections.append((box, str(text), float(conf)))
        return sort_detections(detections)
