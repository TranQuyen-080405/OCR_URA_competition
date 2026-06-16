"""
Tiền xử lý ảnh trước OCR.

Mỗi kỹ thuật là một hàm riêng; pipeline gom ở ``preprocess_single_pass`` /
``generate_multipass_variants``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import (
    PREPROCESS_CENTER_HEIGHT_RATIO,
    PREPROCESS_CENTER_WIDTH_RATIO,
    PREPROCESS_MAX_DIM,
    PREPROCESS_MIN_SIDE_FOR_UPSCALE,
    PREPROCESS_UI_BOTTOM_RATIO,
    PREPROCESS_UI_TOP_RATIO,
    PREPROCESS_UPSCALE_FACTOR,
)


# ---------------------------------------------------------------------------
# Crop / loại UI
# ---------------------------------------------------------------------------


def remove_social_ui_bars(
    img: Image.Image,
    top_ratio: float = PREPROCESS_UI_TOP_RATIO,
    bottom_ratio: float = PREPROCESS_UI_BOTTOM_RATIO,
) -> Image.Image:
    """
    Crop bỏ thanh UI trên/dưới (avatar, like, comment bar) trên screenshot
    TikTok / Facebook dọc.
    """
    if top_ratio <= 0 and bottom_ratio <= 0:
        return img
    w, h = img.size
    y0 = int(h * top_ratio)
    y1 = int(h * (1.0 - bottom_ratio))
    if y1 <= y0 + 8:
        return img
    return img.crop((0, y0, w, y1))


def crop_center_text_region(
    img: Image.Image,
    width_ratio: float = PREPROCESS_CENTER_WIDTH_RATIO,
    height_ratio: float = PREPROCESS_CENTER_HEIGHT_RATIO,
) -> Image.Image:
    """
    Crop vùng giữa — chữ sản phẩm / headline thường nằm ở trung tâm,
    logo/watermark hay ở góc.
    """
    w, h = img.size
    cw = max(8, int(w * width_ratio))
    ch = max(8, int(h * height_ratio))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def upscale_small_image(
    img: Image.Image,
    min_side_target: int = PREPROCESS_MIN_SIDE_FOR_UPSCALE,
    scale_factor: float = PREPROCESS_UPSCALE_FACTOR,
) -> Image.Image:
    """Upscale ảnh nhỏ (cạnh dài < min_side_target) trước OCR."""
    w, h = img.size
    long_side = max(w, h)
    if long_side >= min_side_target:
        return img
    ratio = scale_factor if long_side * scale_factor >= min_side_target else min_side_target / long_side
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return img.resize((new_w, new_h), Image.LANCZOS)


def downscale_large_image(img: Image.Image, max_dim: int = PREPROCESS_MAX_DIM) -> Image.Image:
    """Giảm kích thước nếu cạnh dài vượt max_dim (tránh OOM / chậm)."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    ratio = max_dim / max(w, h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Màu / tương phản
# ---------------------------------------------------------------------------


def to_grayscale(img: Image.Image) -> Image.Image:
    """Chuyển ảnh RGB sang grayscale (PIL L → RGB để EasyOCR đọc được)."""
    return Image.merge("RGB", (img.convert("L"),) * 3)


def apply_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """CLAHE trên kênh L — tăng tương phản cục bộ (chữ mờ / nền phức tạp)."""
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return Image.merge("RGB", (Image.fromarray(enhanced),) * 3)


def apply_adaptive_threshold(
    img: Image.Image,
    block_size: int = 31,
    c: int = 8,
) -> Image.Image:
    """
    Adaptive threshold (Gaussian) — hữu ích khi chữ trắng trên nền tối hoặc ngược lại.
    Trả về RGB 3 kênh cho OCR engine.
    """
    gray = np.array(img.convert("L"))
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = max(3, block)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )
    return Image.merge("RGB", (Image.fromarray(thresh),) * 3)


def sharpen_and_boost_contrast(img: Image.Image, contrast: float = 1.35) -> Image.Image:
    """Tăng contrast nhẹ + sharpen (giữ hành vi baseline cũ)."""
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img.filter(ImageFilter.SHARPEN)


def invert_colors(img: Image.Image) -> Image.Image:
    """Đảo màu — thử khi chữ sáng trên nền tối bị OCR miss."""
    arr = 255 - np.array(img.convert("RGB"))
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------


def deskew_image(img: Image.Image, max_angle: float = 12.0) -> Image.Image:
    """
    Chỉnh góc nghiêng nhẹ (screenshot xiên).
    Dùng minAreaRect trên contour chữ; nếu không ổn định thì trả ảnh gốc.
    """
    gray = np.array(img.convert("L"))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return img

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4 or abs(angle) > max_angle:
        return img

    h, w = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        np.array(img.convert("RGB")),
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)


# ---------------------------------------------------------------------------
# Chuyển đổi / điểm chất lượng
# ---------------------------------------------------------------------------


def pil_to_rgb_numpy(img: Image.Image) -> np.ndarray:
    """PIL RGB → numpy uint8 (H, W, 3)."""
    return np.array(img.convert("RGB"))


def score_ocr_detections(
    detections: list,
    conf_threshold: float,
) -> tuple[float, str]:
    """Điểm chọn variant multi-pass: mean confidence + bonus độ dài text."""
    if not detections:
        return 0.0, ""
    passing = [d for d in detections if d[2] >= conf_threshold]
    if not passing:
        return 0.0, ""
    mean_conf = float(np.mean([d[2] for d in passing]))
    text = " ".join(d[1] for d in sorted(passing, key=lambda r: (r[0][0][1], r[0][0][0])))
    length_bonus = min(len(text) / 120.0, 0.15)
    return mean_conf + length_bonus, text


# alias cũ
score_easyocr_detections = score_ocr_detections


# ---------------------------------------------------------------------------
# Pipeline gom
# ---------------------------------------------------------------------------


def apply_spatial_preprocess(img: Image.Image) -> Image.Image:
    """
    Chuỗi không đổi layout màu: UI crop → center crop → upscale → deskew
    → contrast → downscale.
    """
    img = remove_social_ui_bars(img)
    img = crop_center_text_region(img)
    img = upscale_small_image(img)
    img = deskew_image(img)
    img = sharpen_and_boost_contrast(img)
    img = downscale_large_image(img)
    return img


def build_variant_standard_rgb(img: Image.Image) -> Image.Image:
    """Biến thể 1: RGB sau spatial preprocess (baseline nâng cấp)."""
    return apply_spatial_preprocess(img)


def build_variant_grayscale_clahe(img: Image.Image) -> Image.Image:
    """Biến thể 2: grayscale + CLAHE + contrast."""
    base = apply_spatial_preprocess(img)
    base = apply_clahe(base)
    return sharpen_and_boost_contrast(base, contrast=1.25)


def build_variant_adaptive_threshold(img: Image.Image) -> Image.Image:
    """Biến thể 3: adaptive threshold trên ảnh đã spatial preprocess."""
    base = apply_spatial_preprocess(img)
    return apply_adaptive_threshold(base)


def build_variant_inverted_rgb(img: Image.Image) -> Image.Image:
    """Biến thể 4: đảo màu RGB (chữ sáng / nền tối)."""
    return invert_colors(build_variant_standard_rgb(img))


@dataclass(frozen=True)
class PreprocessVariant:
    name: str
    image: Image.Image


def generate_multipass_variants(img: Image.Image) -> list[PreprocessVariant]:
    """
    Multi-pass: 3–4 biến thể cho OCR chọn theo confidence.
    Thứ tự: standard → CLAHE → adaptive threshold → inverted.
    """
    builders = (
        ("standard_rgb", build_variant_standard_rgb),
        ("grayscale_clahe", build_variant_grayscale_clahe),
        ("adaptive_threshold", build_variant_adaptive_threshold),
        ("inverted_rgb", build_variant_inverted_rgb),
    )
    return [PreprocessVariant(name, fn(img)) for name, fn in builders]


def preprocess_single_pass(img: Image.Image) -> Image.Image:
    """Một pass mặc định (không multi-pass) — đầu vào OCR đơn giản."""
    return build_variant_standard_rgb(img)
