"""Dataset loading, hashing, and splitting for the fraud detection pipeline.

Implements plan §2 (dataset), §5.1 (data quality checks), §7 (splitting),
and the reproducibility requirement from §13 (SHA-256 of the input file
recorded in metadata so retraining on a changed file fails loudly).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("Data/creditcard.csv")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


@dataclass
class Splits:
    """Container for train/val/test splits, kept aligned (X, y) per split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    strategy: Literal["stratified", "time"]
    random_state: int
    fraud_rates: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a small DataFrame describing the splits."""
        rows = []
        for name, X, y in [
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ]:
            rows.append(
                {
                    "split": name,
                    "rows": len(X),
                    "frauds": int(y.sum()),
                    "fraud_rate": float(y.mean()),
                }
            )
        return pd.DataFrame(rows)


def sha256_of_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Stream a SHA-256 of `path` so we never load the 144 MB CSV twice."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(
    path: Path | str = DEFAULT_DATA_PATH,
    drop_duplicates: bool = True,
    verify_shape: bool = True,
) -> pd.DataFrame:
    """Load creditcard.csv and run the plan §5.1 data-quality checks.

    Parameters
    ----------
    path : Path or str
        Location of creditcard.csv. Default Data/creditcard.csv.
    drop_duplicates : bool
        The plan flags ~1,000 exact duplicates and asks for a documented
        decision. We default to dropping them — duplicates inflate train/test
        leakage risk under stratified splitting.
    verify_shape : bool
        Assert the canonical shape (284,807 rows, 492 fraud) before any
        duplicate handling. Disable only for synthetic-data tests.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"creditcard.csv not found at {path}. "
            "Place it in Data/ or pass an explicit path."
        )

    df = pd.read_csv(path)

    # Schema check — fail loudly if the file isn't what we expect.
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"creditcard.csv missing expected columns: {missing}")

    if df.isna().any().any():
        # The Kaggle dataset has no NaNs, but the plan says: "never assume".
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs detected in columns: {bad}")

    if verify_shape:
        if len(df) != EXPECTED_ROWS:
            logger.warning(
                "Row count %d differs from canonical %d", len(df), EXPECTED_ROWS
            )
        n_fraud = int(df[TARGET_COL].sum())
        if n_fraud != EXPECTED_FRAUD:
            logger.warning(
                "Fraud count %d differs from canonical %d", n_fraud, EXPECTED_FRAUD
            )

    n_dupes = int(df.duplicated().sum())
    if drop_duplicates and n_dupes > 0:
        logger.info("Dropping %d exact duplicate rows", n_dupes)
        df = df.drop_duplicates().reset_index(drop=True)

    logger.info(
        "Loaded %d rows, %d fraud (%.4f%%)",
        len(df),
        int(df[TARGET_COL].sum()),
        100.0 * df[TARGET_COL].mean(),
    )
    return df


def split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Splits:
    """70/15/15 stratified split preserving the 0.172% fraud rate per split.

    Done in two steps because sklearn's train_test_split is binary:
        full -> (train+val, test)  then  (train+val) -> (train, val)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Adjust val_size relative to the remaining (train+val) pool.
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, stratify=y_tv, random_state=random_state
    )

    splits = Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        strategy="stratified",
        random_state=random_state,
        fraud_rates={
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    )
    return splits


def split_time_based(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> Splits:
    """Time-ordered split: first 70% of `Time` -> train, next 15% -> val, last 15% -> test.

    Better mimics production deployment, but the late slice will have very
    few fraud cases — the plan calls this out as a deliberate tradeoff worth
    reporting alongside the stratified split (§7.1).
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Invalid train/val fractions")

    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:i_train]
    val_df = df_sorted.iloc[i_train:i_val]
    test_df = df_sorted.iloc[i_val:]

    def xy(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return d[FEATURE_COLS].copy(), d[TARGET_COL].astype(int).copy()

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)

    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        strategy="time",
        random_state=random_state,
        fraud_rates={
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    )


def imbalance_ratio(y: pd.Series) -> float:
    """n_negative / n_positive — used as XGBoost's scale_pos_weight default."""
    pos = int((y == 1).sum())
    if pos == 0:
        raise ValueError("No positive examples in y — cannot compute imbalance ratio")
    return float((y == 0).sum()) / float(pos)


def save_splits(splits: Splits, out_dir: Path | str) -> dict[str, Path]:
    """Persist splits as parquet for fast reload across notebooks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, X, y in [
        ("train", splits.X_train, splits.y_train),
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ]:
        df = X.copy()
        df[TARGET_COL] = y.values
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written[name] = path
    return written


__all__ = [
    "DEFAULT_DATA_PATH",
    "FEATURE_COLS",
    "TARGET_COL",
    "Splits",
    "imbalance_ratio",
    "load_raw",
    "save_splits",
    "sha256_of_file",
    "split_stratified",
    "split_time_based",
]
