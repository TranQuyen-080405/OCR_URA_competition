"""Local composite score (notebook Cell 7)."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.brand_rules import extract_product
from src.config import SOLUTION_PATHS
from src.data_loader import DataBundle
from src.run_logger import RunLogger


def composite_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> float:
    def _clean(val) -> str:
        return "" if pd.isna(val) else str(val).strip()

    required_cols = {"ocr_text", "product_name"}
    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:
        raise ValueError(f"Missing id column: {row_id_column_name}")
    if not required_cols.issubset(solution.columns) or not required_cols.issubset(submission.columns):
        raise ValueError("Both solution and submission must contain ocr_text and product_name")

    sub_ids = submission[row_id_column_name]
    sol_ids = solution[row_id_column_name]
    if sub_ids.duplicated().any():
        raise ValueError("Duplicate image_id in submission")
    if set(sub_ids) != set(sol_ids):
        missing = len(set(sol_ids) - set(sub_ids))
        extra = len(set(sub_ids) - set(sol_ids))
        raise ValueError(f"Submission IDs must match solution exactly (missing {missing}, extra {extra})")

    merged = solution.merge(submission, on=row_id_column_name, suffixes=("_gt", "_pred"), how="inner")
    if merged.empty:
        raise ValueError("No matching rows after merge")

    def token_f1(gt: str, pred: str) -> float:
        gt = _clean(gt)
        pred = _clean(pred)
        if not gt and not pred:
            return 1.0
        gt_tokens = set(gt.lower().split())
        pred_tokens = set(pred.lower().split())
        if not gt_tokens or not pred_tokens:
            return 0.0
        common = gt_tokens & pred_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def cer(gt: str, pred: str) -> float:
        gt = _clean(gt)
        pred = _clean(pred)
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        m, n = len(gt), len(pred)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if gt[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return min(dp[n] / len(gt), 1.0)

    product_f1 = merged.apply(lambda r: token_f1(r["product_name_gt"], r["product_name_pred"]), axis=1).mean()
    avg_cer = merged.apply(lambda r: cer(r["ocr_text_gt"], r["ocr_text_pred"]), axis=1).mean()
    return round(float(0.6 * product_f1 + 0.4 * (1.0 - avg_cer)), 4)


def _load_solution() -> pd.DataFrame | None:
    for path in SOLUTION_PATHS:
        if path.exists():
            return pd.read_csv(path, keep_default_na=False)
    return None


def run_evaluation(
    data: DataBundle,
    results: list[dict],
    predict_product: Callable[[str], str],
    logger: RunLogger,
) -> None:
    logger.section("Cell 7 — Local Evaluation (Composite Score)")
    logger.log("Using standalone inline composite score")

    solution_df = _load_solution()
    if solution_df is None:
        logger.log("solution.csv not found — skip local scoring (OK on Kaggle participant side)")
        return

    logger.log(f"Loaded solution ({len(solution_df):,} rows)")

    if not results:
        logger.log("No OCR predictions yet — running oracle ablation on GT ocr_text (product head only)\n")
        gt = solution_df[["image_id", "ocr_text", "product_name"]].copy()
        for col in ("ocr_text", "product_name"):
            gt[col] = gt[col].astype(str).str.strip()
        sub_rules = gt.copy()
        sub_rules["product_name"] = sub_rules["ocr_text"].map(extract_product)
        sub_train = gt.copy()
        sub_train["product_name"] = sub_train["ocr_text"].map(predict_product)
        logger.log(f"  Rules-only product:     {composite_score(gt, sub_rules):.4f}")
        logger.log(f"  Train-enhanced product: {composite_score(gt, sub_train):.4f}")
        logger.log("\nRun full pipeline for OCR + product model score")
        return

    pred_df = pd.DataFrame(results)[["image_id", "ocr_text", "product_name"]]
    pred_rules = pred_df.copy()
    pred_rules["product_name"] = pred_rules["ocr_text"].map(extract_product)

    gt = solution_df[["image_id", "ocr_text", "product_name"]]
    s_full = composite_score(gt, pred_df)
    s_rules = composite_score(gt, pred_rules)

    logger.log("\nComposite score (full pipeline — OCR + train product model):")
    logger.log(f"  Score: {s_full:.4f}")

    logger.log("\nAblation (same OCR, rules-only product):")
    logger.log(f"  Score: {s_rules:.4f}")

    if "Usage" in solution_df.columns:
        for split in ("Public", "Private"):
            ids = set(solution_df.loc[solution_df["Usage"] == split, "image_id"])
            if not ids:
                continue
            gt_s = gt[gt["image_id"].isin(ids)]
            pred_s = pred_df[pred_df["image_id"].isin(ids)]
            logger.log(f"  {split}: {composite_score(gt_s, pred_s):.4f}")

    sample_empty = pd.read_csv(data.sample_csv, keep_default_na=False)
    sample_empty["ocr_text"] = ""
    sample_empty["product_name"] = ""
    logger.log(f"\nReference — empty sample_submission: {composite_score(gt, sample_empty):.4f}")
