"""Training pipeline: Optuna tuning -> refit on train+val -> single test eval.

Implements plan §8 (imbalance handling), §9 (model selection & tuning),
and the §13 reproducibility requirements (seeds, dataset hash, library
versions, exact pip versions). Saves the fitted sklearn Pipeline plus a
metadata.json sidecar to ./models/.

Run inside Docker:
    docker compose --profile train up train

Or directly:
    python -m src.train --trials 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import src
from src.data import (
    DEFAULT_DATA_PATH,
    Splits,
    imbalance_ratio,
    load_raw,
    sha256_of_file,
    split_stratified,
)
from src.evaluate import evaluate
from src.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "champion.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed numpy, random, and ensure xgboost gets a fixed random_state.

    Plan §13 explicitly requires seeding everything. xgboost's RNG is
    seeded per-estimator via random_state on each XGBClassifier instance
    rather than globally — handled in build_pipeline().
    """
    random.seed(seed)
    np.random.seed(seed)


def library_versions() -> dict[str, str]:
    """Snapshot of every library that affects model output for metadata.json."""
    import imblearn  # noqa: WPS433
    import lightgbm  # noqa: WPS433
    import shap  # noqa: WPS433
    import sklearn  # noqa: WPS433

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "lightgbm": lightgbm.__version__,
        "imbalanced-learn": imblearn.__version__,
        "optuna": optuna.__version__,
        "shap": shap.__version__,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_pipeline(
    params: dict,
    scale_pos_weight: float,
    random_state: int = 42,
) -> Pipeline:
    """sklearn Pipeline = FeatureEngineer -> XGBClassifier.

    XGBoost is the primary candidate per plan §9.1; tree models don't
    need scaling (§6.4) so the pipeline is just two steps.
    """
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Return an Optuna objective that maximizes mean PR-AUC over StratifiedKFold."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-2, 3e-1, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores: list[float] = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pipe = build_pipeline(
                params, scale_pos_weight=scale_pos_weight, random_state=random_state
            )
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)[:, 1]
            fold_scores.append(float(average_precision_score(y_va, proba)))

        # Optuna maximizes the returned value; return mean PR-AUC and report
        # std as a user attribute so high-variance trials are visible.
        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        trial.set_user_attr("pr_auc_std", std)
        trial.set_user_attr("fold_scores", fold_scores)
        return mean

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    data_path: Path = DEFAULT_DATA_PATH,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    out_dir: Path = MODEL_DIR,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> dict:
    """End-to-end training run. Returns the metadata dict that gets written."""
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("Loading data from %s", data_path)
    df = load_raw(data_path)
    data_hash = sha256_of_file(Path(data_path))
    logger.info("Dataset SHA-256: %s", data_hash)

    logger.info("Stratified 70/15/15 split with seed=%d", seed)
    splits: Splits = split_stratified(df, random_state=seed)
    logger.info("Splits:\n%s", splits.summary().to_string(index=False))

    spw = imbalance_ratio(splits.y_train)
    logger.info("scale_pos_weight (n_neg/n_pos on train) = %.2f", spw)

    # Optuna study on TRAIN ONLY using StratifiedKFold (plan §9.4).
    logger.info("Optuna study: %d trials, TPE sampler, %d-fold PR-AUC", n_trials, n_splits)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(
            splits.X_train, splits.y_train, spw, n_splits=n_splits, random_state=seed
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = dict(study.best_trial.params)
    cv_pr_auc_mean = float(study.best_trial.value)
    cv_pr_auc_std = float(study.best_trial.user_attrs.get("pr_auc_std", float("nan")))
    logger.info(
        "Best CV PR-AUC: %.4f (+/- %.4f) | params: %s",
        cv_pr_auc_mean,
        cv_pr_auc_std,
        best_params,
    )

    # Refit on train+val (plan §9.5).
    logger.info("Refitting champion on train + val (%d rows)", len(splits.X_train) + len(splits.X_val))
    X_fit = pd.concat([splits.X_train, splits.X_val], axis=0)
    y_fit = pd.concat([splits.y_train, splits.y_val], axis=0)
    champion = build_pipeline(best_params, scale_pos_weight=spw, random_state=seed)
    champion.fit(X_fit, y_fit)

    # Single-shot test evaluation (plan §9.5: cardinal sin to iterate).
    logger.info("Evaluating champion ONCE on the held-out test set")
    proba_test = champion.predict_proba(splits.X_test)[:, 1]
    report = evaluate(
        splits.y_test.to_numpy(),
        proba_test,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    logger.info(
        "Test PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f recall@1%%FPR=%.4f",
        report.pr_auc,
        report.roc_auc,
        report.brier,
        report.recall_at_1pct_fpr,
    )

    # Persist artefacts.
    model_path = out_dir / "champion.pkl"
    joblib.dump(champion, model_path)
    logger.info("Wrote model -> %s", model_path)

    # Choose the cost-optimal threshold as the deployed default; the plan
    # says default 0.5 is almost always wrong on imbalanced data (§10.2).
    chosen_threshold = report.thresholds["cost_optimal"].threshold

    metadata = {
        "model_version": src.__version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trained_at_unix": int(time.time()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "data": {
            "path": str(data_path),
            "sha256": data_hash,
            "n_rows": int(len(df)),
            "n_fraud": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()),
        },
        "split": {
            "strategy": splits.strategy,
            "random_state": splits.random_state,
            "fraud_rates": splits.fraud_rates,
            "sizes": {
                "train": int(len(splits.X_train)),
                "val": int(len(splits.X_val)),
                "test": int(len(splits.X_test)),
            },
        },
        "model": {
            "estimator": "xgboost.XGBClassifier",
            "scale_pos_weight": spw,
            "best_params": best_params,
            "cv_pr_auc_mean": cv_pr_auc_mean,
            "cv_pr_auc_std": cv_pr_auc_std,
            "n_optuna_trials": n_trials,
            "n_cv_splits": n_splits,
        },
        "test_metrics": report.as_dict(),
        "deployment": {
            "chosen_threshold": chosen_threshold,
            "threshold_strategy": "cost_optimal",
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "feature_names_out": list(
                champion.named_steps["features"].feature_names_out_
            ),
        },
        "library_versions": library_versions(),
        "seed": seed,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Wrote metadata -> %s", metadata_path)
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fraud detection champion model")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--trials", type=int, default=50, help="Optuna trial budget")
    p.add_argument("--folds", type=int, default=5, help="StratifiedKFold splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=MODEL_DIR)
    p.add_argument("--cost-fn", type=float, default=100.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    run(
        data_path=args.data,
        n_trials=args.trials,
        n_splits=args.folds,
        seed=args.seed,
        out_dir=args.out,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )
