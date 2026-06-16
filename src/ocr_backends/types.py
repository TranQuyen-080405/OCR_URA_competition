"""Shared OCR detection types."""

from __future__ import annotations

# box: list of 4 [x,y] points; text; confidence
OcrDetection = tuple[list, str, float]


def sort_detections(detections: list[OcrDetection]) -> list[OcrDetection]:
    return sorted(detections, key=lambda r: (r[0][0][1], r[0][0][0]))


def detections_to_text(detections: list[OcrDetection], conf_threshold: float) -> str:
    lines = [text for _, text, conf in detections if conf > conf_threshold]
    return " ".join(lines)
