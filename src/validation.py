"""Validate and export submission.csv (notebook Cell 6)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.data_loader import DataBundle
from src.run_logger import RunLogger


def write_submission_csv(df: pd.DataFrame, path: Path) -> None:
    out = df[["image_id", "ocr_text", "product_name"]].copy()
    for col in ("ocr_text", "product_name"):
        out[col] = out[col].fillna("").astype(str).str.strip()
        out.loc[out[col] == "", col] = " "
    out.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def validate_and_export(
    data: DataBundle,
    results: list[dict],
    logger: RunLogger,
) -> pd.DataFrame:
    logger.section("Cell 6 — Validate and Export submission.csv")

    sub = pd.DataFrame(results)[["image_id", "ocr_text", "product_name"]]
    sample = pd.read_csv(data.sample_csv, keep_default_na=False)

    logger.log("Validating submission format...\n")
    checks: dict[str, bool] = {}

    expected_ids = set(sample["image_id"])
    got_ids = set(sub["image_id"])
    checks["AC-1 Row count match"] = len(sub) == len(sample)
    checks["AC-2 No extra IDs"] = len(got_ids - expected_ids) == 0
    checks["AC-3 No missing IDs"] = len(expected_ids - got_ids) == 0
    checks["AC-4 No duplicate IDs"] = not sub["image_id"].duplicated().any()
    checks["AC-5 No null values"] = not sub.isnull().any().any()
    checks["AC-6 No newline in text"] = not sub["ocr_text"].str.contains(r"\n|\t", regex=True, na=False).any()
    checks["AC-7 Columns correct"] = list(sub.columns) == ["image_id", "ocr_text", "product_name"]

    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        logger.log(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    logger.log("")
    if not all_pass:
        logger.log("Validation failed — fix issues above before submitting.")
        return sub

    sub = sub.set_index("image_id").reindex(sample["image_id"]).reset_index()
    sub["ocr_text"] = sub["ocr_text"].fillna("").astype(str).str.strip()
    sub["product_name"] = sub["product_name"].fillna("").astype(str).str.strip()
    write_submission_csv(sub, data.output_csv)

    logger.log("All checks passed.")
    logger.log(f"Saved to: {data.output_csv}")
    logger.log("\nFirst 5 rows:")
    logger.log(sub.head().to_string())
    ocr_fill = (sub["ocr_text"].str.strip() != "").mean()
    prod_fill = (sub["product_name"].str.strip() != "").mean()
    logger.log(f"\nStats: OCR fill={ocr_fill:.1%} | Product fill={prod_fill:.1%} | Rows={len(sub):,}")
    logger.log("\nNext steps:")
    logger.log("  1. Kaggle -> Competition -> Submit Predictions")
    logger.log("  2. Upload submission.csv")
    logger.log("  3. Check score on the Public Leaderboard")

    return sub
