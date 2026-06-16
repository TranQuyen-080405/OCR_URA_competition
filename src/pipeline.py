"""Main OCR loop with checkpointing (notebook Cell 5)."""

from __future__ import annotations

import gc

import pandas as pd
from tqdm import tqdm

from src.config import SAVE_EVERY
from src.data_loader import DataBundle
from src.ocr_engine import OcrEngine
from src.run_logger import RunLogger


def run_pipeline(
    data: DataBundle,
    engine: OcrEngine,
    logger: RunLogger,
    *,
    max_images: int | None = None,
    use_checkpoint: bool = True,
) -> list[dict]:
    logger.section("Cell 5 — Main Loop (OCR + product)")

    done_ids: set[str] = set()
    results: list[dict] = []

    if use_checkpoint and data.checkpoint_csv.exists():
        ckpt = pd.read_csv(data.checkpoint_csv, keep_default_na=False)
        done_ids = set(ckpt["image_id"])
        results = ckpt.to_dict("records")
        logger.log(f"Resuming from checkpoint: {len(done_ids):,} images done")
    else:
        logger.log("Starting fresh")

    pending = [i for i in data.test_df["image_id"] if i not in done_ids]
    if max_images is not None:
        pending = pending[:max_images]
        logger.log(f"Limiting to first {max_images} pending images (--max-images)")

    logger.log(f"Pending: {len(pending):,} | Done: {len(done_ids):,}")
    logger.log("")

    errors = 0
    for idx, img_id in enumerate(tqdm(pending, desc="OCR Progress")):
        ocr_text, product_name = engine.run_ocr(img_id)

        if ocr_text == "" and (data.images_dir / f"{img_id}.jpg").exists():
            errors += 1

        results.append({
            "image_id": img_id,
            "ocr_text": ocr_text,
            "product_name": product_name,
        })

        if (idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(data.checkpoint_csv, index=False, encoding="utf-8")
            gc.collect()

    pd.DataFrame(results).to_csv(data.checkpoint_csv, index=False, encoding="utf-8")

    df_result = pd.DataFrame(results)
    ocr_fill = (df_result["ocr_text"].str.strip() != "").mean()
    prod_fill = (df_result["product_name"].str.strip() != "").mean()

    logger.log("")
    logger.log(f"Processed     : {len(df_result):,}")
    logger.log(f"OCR fill rate : {ocr_fill:.1%}")
    logger.log(f"Product fill  : {prod_fill:.1%}")
    logger.log(f"OCR failures  : {errors:,}")

    return results
