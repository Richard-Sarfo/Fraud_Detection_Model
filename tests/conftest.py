"""Shared fixtures for the test suite.

Generates a small synthetic dataset that mimics the creditcard.csv
schema so tests run without depending on the 144 MB Kaggle file.
The notebook-grade data is too big for CI; tests must be deterministic
and quick.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.features import FeatureEngineer

V_COLS = [f"V{i}" for i in range(1, 29)]
COLUMNS = ["Time"] + V_COLS + ["Amount", "Class"]


def _make_synthetic(
    n_genuine: int = 2_000,
    n_fraud: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a tiny synthetic dataset that preserves the canonical schema.

    Fraud rows are drawn from a slightly shifted distribution on V14, V12,
    V10 (the §5.2 most-discriminative features) so a model trained on this
    can distinguish the two classes — useful for end-to-end pipeline tests.
    """
    rng = np.random.default_rng(seed)

    n = n_genuine + n_fraud
    times = rng.uniform(0, 172_800, size=n).astype(float)  # 2-day window like the real set
    times.sort()

    v_genuine = rng.normal(loc=0, scale=1.0, size=(n_genuine, 28))
    # Shift V14, V12, V10, V17, V4 for fraud rows to make them learnable.
    shifts = np.zeros(28)
    for col in [14, 12, 10, 17, 4]:
        shifts[col - 1] = -3.0
    v_fraud = rng.normal(loc=shifts, scale=1.0, size=(n_fraud, 28))
    v_all = np.vstack([v_genuine, v_fraud])

    amount_genuine = rng.gamma(shape=1.5, scale=60.0, size=n_genuine)
    amount_fraud = rng.gamma(shape=1.0, scale=30.0, size=n_fraud)
    amount_all = np.concatenate([amount_genuine, amount_fraud])

    classes = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(n_fraud, dtype=int)])
    perm = rng.permutation(n)

    df = pd.DataFrame(v_all[perm], columns=V_COLS)
    df.insert(0, "Time", times[perm])
    df["Amount"] = amount_all[perm]
    df["Class"] = classes[perm]
    return df[COLUMNS]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Tiny synthetic dataset that follows creditcard.csv's schema."""
    return _make_synthetic()


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, synthetic_df) -> Path:
    """Persist the synthetic frame as a CSV for load_raw() round-trip tests."""
    out = tmp_path_factory.mktemp("data") / "creditcard.csv"
    synthetic_df.to_csv(out, index=False)
    return out


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_df) -> Pipeline:
    """Quick-to-train Pipeline (FeatureEngineer + XGBoost) on synthetic data.

    Used by the API contract tests so they don't need to load a real
    champion.pkl produced by a 50-trial Optuna run.
    """
    X = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"].astype(int)
    pipe = Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("model", xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            n_estimators=80, max_depth=4, learning_rate=0.2,
            scale_pos_weight=float((y == 0).sum() / max(int((y == 1).sum()), 1)),
            tree_method="hist", random_state=0, n_jobs=1,
        )),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def model_artefacts(tmp_path: Path, fitted_pipeline: Pipeline) -> dict:
    """Drop a champion.pkl + metadata.json into tmp_path so the API can load them."""
    model_path = tmp_path / "champion.pkl"
    meta_path = tmp_path / "metadata.json"
    joblib.dump(fitted_pipeline, model_path)
    meta_path.write_text(json.dumps({
        "model_version": "0.1.0",
        "deployment": {
            "chosen_threshold": 0.5,
            "threshold_strategy": "default",
            "cost_fn": 100.0,
            "cost_fp": 1.0,
            "feature_names_out": list(
                fitted_pipeline.named_steps["features"].feature_names_out_
            ),
        },
    }))
    return {"model_path": model_path, "metadata_path": meta_path}
