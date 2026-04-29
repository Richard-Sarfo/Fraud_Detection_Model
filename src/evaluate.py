"""Metrics, threshold selection, and bootstrapped confidence intervals.

Implements plan §10 (evaluation & threshold selection): PR-AUC as the
headline metric under heavy class imbalance, ROC-AUC reported only for
comparison, recall-at-fixed-FPR and precision-at-fixed-recall as
operational views, F1/F2 at chosen thresholds, Brier score for
calibration, and three threshold strategies (default 0.5, F1-optimal,
cost-optimal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdReport:
    """Metrics computed at a single decision threshold."""

    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


@dataclass
class EvaluationReport:
    """Threshold-independent metrics plus per-threshold reports."""

    pr_auc: float
    roc_auc: float
    brier: float
    recall_at_1pct_fpr: float
    precision_at_80pct_recall: float
    thresholds: dict[str, ThresholdReport] = field(default_factory=dict)
    n_samples: int = 0
    n_positives: int = 0

    def as_dict(self) -> dict:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "recall_at_1pct_fpr": self.recall_at_1pct_fpr,
            "precision_at_80pct_recall": self.precision_at_80pct_recall,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "thresholds": {k: v.as_dict() for k, v in self.thresholds.items()},
        }


def _validate(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_score {y_score.shape} shape mismatch"
        )
    if y_true.size == 0:
        raise ValueError("Empty y_true / y_score")
    if y_true.sum() == 0:
        raise ValueError("No positive examples in y_true — cannot evaluate fraud model")
    return y_true, y_score


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.01
) -> float:
    """Best recall achievable at FPR <= `max_fpr` (plan §10.1)."""
    y_true, y_score = _validate(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Largest TPR among thresholds with FPR <= max_fpr.
    mask = fpr <= max_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, min_recall: float = 0.80
) -> float:
    """Best precision achievable while keeping recall >= `min_recall`."""
    y_true, y_score = _validate(y_true, y_score)
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = r >= min_recall
    if not mask.any():
        return 0.0
    return float(p[mask].max())


def find_threshold_max_f1(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """Threshold that maximizes F1 on the PR curve. Returns (threshold, f1)."""
    y_true, y_score = _validate(y_true, y_score)
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one fewer threshold than p/r points.
    p, r = p[:-1], r[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)
    if f1.size == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(t[idx]), float(f1[idx])


def find_threshold_min_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> tuple[float, float]:
    """Cost-optimal threshold (plan §10.2).

    Defaults model the typical fraud business: a missed fraud (false
    negative) is worth ~$100 in chargebacks/refunds while a false alarm
    is worth ~$1 of customer-friction cost. Override these to match a
    real business.
    """
    y_true, y_score = _validate(y_true, y_score)
    # Scan thresholds at unique score values plus 0 and 1.
    candidates = np.unique(np.concatenate([y_score, [0.0, 1.0]]))
    best_cost = np.inf
    best_t = 0.5
    for t in candidates:
        pred = (y_score >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        total = cost_fn * fn + cost_fp * fp
        if total < best_cost:
            best_cost = total
            best_t = float(t)
    return best_t, float(best_cost)


def threshold_report(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, name: str
) -> ThresholdReport:
    y_true, y_score = _validate(y_true, y_score)
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return ThresholdReport(
        name=name,
        threshold=float(threshold),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        f2=float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
    )


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[str, float] | None = None,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> EvaluationReport:
    """Compute the full plan §10 metric suite at a configurable set of thresholds.

    If `thresholds` is None, defaults to {default: 0.5, f1_optimal:
    derived, cost_optimal: derived} so a single call yields the
    three-threshold view the plan asks for in §10.2.
    """
    y_true, y_score = _validate(y_true, y_score)

    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    r_at_fpr = recall_at_fpr(y_true, y_score, max_fpr=0.01)
    p_at_r = precision_at_recall(y_true, y_score, min_recall=0.80)

    if thresholds is None:
        f1_t, _ = find_threshold_max_f1(y_true, y_score)
        cost_t, _ = find_threshold_min_cost(
            y_true, y_score, cost_fn=cost_fn, cost_fp=cost_fp
        )
        thresholds = {
            "default": 0.5,
            "f1_optimal": f1_t,
            "cost_optimal": cost_t,
        }

    reports = {
        name: threshold_report(y_true, y_score, t, name)
        for name, t in thresholds.items()
    }

    return EvaluationReport(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=brier,
        recall_at_1pct_fpr=r_at_fpr,
        precision_at_80pct_recall=p_at_r,
        thresholds=reports,
        n_samples=int(y_true.size),
        n_positives=int(y_true.sum()),
    )


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Literal["pr_auc", "roc_auc"] = "pr_auc",
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a metric on the held-out set (plan §10.4).

    Returns (point_estimate, lower_95, upper_95). Stratified by class
    so a 0.172%-positive test set never produces a bootstrap sample
    with zero positives.
    """
    y_true, y_score = _validate(y_true, y_score)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    point = float(fn(y_true, y_score))

    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        samples[i] = fn(y_true[idx], y_score[idx])

    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


__all__ = [
    "EvaluationReport",
    "ThresholdReport",
    "bootstrap_metric",
    "evaluate",
    "find_threshold_max_f1",
    "find_threshold_min_cost",
    "precision_at_recall",
    "recall_at_fpr",
    "threshold_report",
]
