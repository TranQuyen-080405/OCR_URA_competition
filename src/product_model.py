"""Lightweight product classifier (notebook Cell 3b)."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.brand_rules import extract_product
from src.run_logger import RunLogger


class ProductPredictor:
    def __init__(self, min_class_count: int = 3, prob_threshold: float = 0.60, max_features: int = 3000):
        self.min_class_count = min_class_count
        self.prob_threshold = prob_threshold
        self.max_features = max_features
        self._has_clf = self._prod_clf = None
        self._n_train = self._n_classes = 0
        self._rule_fn: Callable[[str], str] = extract_product

    def fit(self, train_labels: pd.DataFrame, rule_fn: Callable[[str], str]) -> ProductPredictor:
        df = train_labels.copy()
        df["ocr_text"] = df["ocr_text"].astype(str).str.strip()
        df["product_name"] = df["product_name"].astype(str).str.strip()
        self._rule_fn = rule_fn
        self._has_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ])
        self._has_clf.fit(df["ocr_text"], (df["product_name"] != "").astype(int))
        pos = df[(df["ocr_text"] != "") & (df["product_name"] != "")]
        keep = pos["product_name"].value_counts()
        keep = keep[keep >= self.min_class_count].index
        pos = pos[pos["product_name"].isin(keep)]
        self._prod_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ])
        if len(pos):
            self._prod_clf.fit(pos["ocr_text"], pos["product_name"])
        self._n_train, self._n_classes = len(df), pos["product_name"].nunique() if len(pos) else 0
        return self

    def predict(self, ocr_text: str) -> str:
        ocr_text = "" if ocr_text is None else str(ocr_text).strip()
        if not ocr_text:
            return ""
        ruled = self._rule_fn(ocr_text)
        if ruled:
            return ruled
        if self._has_clf is None or self._prod_clf is None:
            return ""
        proba = self._has_clf.predict_proba([ocr_text])[0]
        if 1 not in self._has_clf.classes_ or proba[list(self._has_clf.classes_).index(1)] < self.prob_threshold:
            return ""
        return str(self._prod_clf.predict([ocr_text])[0])


def build_product_predictor(
    train_labels_df: pd.DataFrame | None,
    logger: RunLogger,
) -> Callable[[str], str]:
    logger.section("Cell 3b — Train Lightweight Product Head")

    product_predictor: ProductPredictor | None = None
    if train_labels_df is not None:
        product_predictor = ProductPredictor(min_class_count=3, prob_threshold=0.60, max_features=3000)
        product_predictor.fit(train_labels_df, extract_product)
        pos = train_labels_df.copy()
        pos["ocr_text"] = pos["ocr_text"].astype(str).str.strip()
        pos["product_name"] = pos["product_name"].astype(str).str.strip()
        n_pairs = ((pos["ocr_text"] != "") & (pos["product_name"] != "")).sum()
        logger.log(f"Trained lightweight product head on {len(train_labels_df):,} rows ({n_pairs:,} OCR+product pairs)")
    else:
        logger.log("train_labels.csv not found — rules-only mode (simplest lightweight path)")

    def predict_product(ocr_text: str) -> str:
        if product_predictor is not None:
            return product_predictor.predict(ocr_text)
        return extract_product(ocr_text)

    return predict_product
