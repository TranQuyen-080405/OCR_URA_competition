"""Load dataset paths and CSVs (notebook Cell 2)."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    IMAGE_DIR_CANDIDATES,
    IMAGE_ZIP_NAMES,
    LOCAL_DATASET_ROOTS,
    TRAIN_LABELS_CANDIDATES,
)
from src.run_logger import RunLogger


@dataclass
class DataBundle:
    input_dir: Path
    work_dir: Path
    images_dir: Path
    test_df: pd.DataFrame
    train_labels_df: pd.DataFrame | None
    test_csv: Path
    sample_csv: Path
    output_csv: Path
    checkpoint_csv: Path
    image_ids_on_disk: set[str]


def _input_roots() -> list[Path]:
    roots: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        comp_root = kaggle_input / "competitions"
        if comp_root.exists():
            roots.extend(sorted(comp_root.iterdir()))
        roots.extend(sorted(kaggle_input.iterdir()))
    for local in LOCAL_DATASET_ROOTS:
        if local.exists():
            roots.append(local.resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def resolve_input_dir() -> Path:
    for root in _input_roots():
        if (root / "test.csv").exists():
            return root
    for search_root in LOCAL_DATASET_ROOTS:
        if not search_root.exists():
            continue
        for test_csv in sorted(search_root.rglob("test.csv")):
            return test_csv.parent
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for test_csv in sorted(kaggle_input.rglob("test.csv")):
            return test_csv.parent
    hint = (
        "Dataset not found. On Kaggle: Add Input → Competition Data → "
        "The 2nd URA Hackathon (expect test.csv + test_images/ under "
        "/kaggle/input/competitions/the-2nd-ura-hackathon/). "
        "Locally: place competition files under the-2nd-ura-hackathon/ "
        "(test.csv, sample_submission.csv, test_images/images/*.jpg)."
    )
    if kaggle_input.exists():
        listing = sorted(str(p) for p in kaggle_input.rglob("*") if p.is_file())[:20]
        hint += f"\n/kaggle/input files (first 20): {listing}"
    raise FileNotFoundError(hint)


def _find_images_dir(input_dir: Path) -> Path | None:
    for rel in IMAGE_DIR_CANDIDATES:
        images_dir = input_dir / rel
        if images_dir.is_dir() and any(images_dir.glob("*.jpg")):
            return images_dir
    return None


def _find_images_zip(input_dir: Path) -> Path | None:
    for rel in IMAGE_ZIP_NAMES:
        zip_path = input_dir / rel
        if zip_path.is_file():
            return zip_path
    for zip_path in sorted(input_dir.rglob("*.zip")):
        name = zip_path.name.lower()
        if "train" in name and "test" not in name:
            continue
        if name in ("test_images.zip", "images.zip") or name.endswith("images.zip"):
            return zip_path
    return None


def setup_images_dir(input_dir: Path, work_dir: Path, logger: RunLogger) -> Path:
    images_dir = _find_images_dir(input_dir)
    if images_dir is not None:
        return images_dir

    zip_path = _find_images_zip(input_dir)
    if zip_path is None:
        raise FileNotFoundError(
            f"No test images under {input_dir}. Expected test_images/, images/, "
            f"or one of {IMAGE_ZIP_NAMES}."
        )

    extract_root = work_dir / "dataset_images"
    images_dir = extract_root / "images"
    if not any(images_dir.glob("*.jpg")):
        extract_root.mkdir(parents=True, exist_ok=True)
        logger.log(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        if not any(images_dir.glob("*.jpg")):
            for rel in ("test_images/images", "test_images", "images"):
                alt = extract_root / rel
                if alt.is_dir() and any(alt.glob("*.jpg")):
                    images_dir = alt
                    break
    return images_dir


def _load_train_labels(input_dir: Path) -> pd.DataFrame | None:
    candidates = [
        input_dir / "train_labels.csv",
        Path("the-2nd-ura-hackathon/train_labels.csv"),
        Path("public_dataset/train_labels.csv"),
    ]
    seen: set[Path] = set()
    for labels_path in candidates:
        if labels_path in seen:
            continue
        seen.add(labels_path)
        if labels_path.exists():
            return pd.read_csv(labels_path, keep_default_na=False)
    return None


def load_data(logger: RunLogger) -> DataBundle:
    logger.section("Cell 2 — Load Data")

    input_dir = resolve_input_dir()
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    images_dir = setup_images_dir(input_dir, work_dir, logger)

    test_csv = input_dir / "test.csv"
    sample_csv = input_dir / "sample_submission.csv"
    output_csv = work_dir / "submission.csv"
    checkpoint_csv = work_dir / "checkpoint.csv"

    test_df = pd.read_csv(test_csv)
    image_ids_on_disk = {p.stem for p in images_dir.glob("*.jpg")}
    train_labels_df = _load_train_labels(input_dir)

    logger.log(f"Input dir   : {input_dir}")
    logger.log(f"Test set    : {len(test_df):,} images")
    logger.log(f"Images dir  : {images_dir} ({len(image_ids_on_disk):,} jpg)")
    logger.log(f"Working dir : {work_dir}")

    if train_labels_df is not None:
        ocr_fill = (train_labels_df["ocr_text"].astype(str).str.strip() != "").mean()
        prod_fill = (train_labels_df["product_name"].astype(str).str.strip() != "").mean()
        logger.log(
            f"Train labels: {len(train_labels_df):,} rows "
            f"(draft v1 | OCR {ocr_fill:.1%} | product {prod_fill:.1%})"
        )
        logger.log("  Use train.csv + train_images.zip with train_labels_df for supervised training.")
    else:
        logger.log("Train labels: not found (test-only mode)")

    missing = set(test_df["image_id"]) - image_ids_on_disk
    if missing:
        logger.log(f"Warning: {len(missing)} image_ids have no jpg (OCR will return empty for those)")
    else:
        logger.log("All test image_ids have matching jpg files")

    logger.log("\nPreview test.csv:")
    logger.log(test_df.head(3).to_string())

    return DataBundle(
        input_dir=input_dir,
        work_dir=work_dir,
        images_dir=images_dir,
        test_df=test_df,
        train_labels_df=train_labels_df,
        test_csv=test_csv,
        sample_csv=sample_csv,
        output_csv=output_csv,
        checkpoint_csv=checkpoint_csv,
        image_ids_on_disk=image_ids_on_disk,
    )
