"""SHAP wrappers for global and local explainability.

Implements plan §11. Designed to be cheap enough to run on every API
request: TreeExplainer is exact for XGBoost, the explainer is built
once at startup, and per-row SHAP costs ~10-30 ms for a 30-feature
model — within the 50 ms p99 budget from §1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


@dataclass
class FeatureContribution:
    """A single feature's contribution to a prediction (used by the API)."""

    name: str
    value: float
    shap_value: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": float(self.value),
            "shap_value": float(self.shap_value),
        }


class FraudExplainer:
    """Wraps a fitted Pipeline so callers don't have to know about FE state.

    The training pipeline is `FeatureEngineer -> XGBClassifier`. SHAP's
    TreeExplainer expects the post-FE numeric matrix, so this class
    runs the FE step explicitly and feeds the result to the explainer.
    """

    def __init__(self, pipeline: Pipeline):
        if "features" not in pipeline.named_steps:
            raise ValueError("Pipeline missing 'features' step")
        if "model" not in pipeline.named_steps:
            raise ValueError("Pipeline missing 'model' step")

        self.pipeline = pipeline
        self.feature_engineer = pipeline.named_steps["features"]
        self.model = pipeline.named_steps["model"]
        self.feature_names = list(self.feature_engineer.feature_names_out_)
        # TreeExplainer is exact and fast for XGBoost (plan §11.4).
        self.explainer = shap.TreeExplainer(self.model)

    # ---- core API -------------------------------------------------------

    def shap_values(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Return per-row, per-feature SHAP values in the post-FE feature space.

        Shape: (n_rows, n_features). Caller can sum to get the model's
        log-odds shift relative to the base value.
        """
        X_fe = self.feature_engineer.transform(X_raw)
        sv = self.explainer.shap_values(X_fe)
        # SHAP returns either a 2D array (binary) or a list of arrays
        # (older versions). Normalize to a 2D ndarray.
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        return np.asarray(sv)

    def expected_value(self) -> float:
        """Model's base value in log-odds space."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[-1]
        return float(ev)

    def top_contributors(
        self,
        X_raw: pd.DataFrame,
        k: int = 3,
    ) -> list[list[FeatureContribution]]:
        """Top-k features (by |SHAP|) per row.

        Returns a list of length n_rows; each element is a list of k
        FeatureContribution dicts ordered by descending |shap_value|.
        Used by the API to populate the `top_features` response field
        per plan §12.3.
        """
        X_fe = self.feature_engineer.transform(X_raw)
        sv = self.shap_values(X_raw)
        out: list[list[FeatureContribution]] = []
        names = np.asarray(self.feature_names)
        values = X_fe.to_numpy()
        for row_idx in range(sv.shape[0]):
            order = np.argsort(np.abs(sv[row_idx]))[::-1][:k]
            row_out = [
                FeatureContribution(
                    name=str(names[i]),
                    value=float(values[row_idx, i]),
                    shap_value=float(sv[row_idx, i]),
                )
                for i in order
            ]
            out.append(row_out)
        return out

    # ---- plotting helpers (notebook-side) -------------------------------

    def summary_plot(
        self,
        X_raw: pd.DataFrame,
        sample: int | None = 10_000,
        seed: int = 42,
        plot_type: str = "dot",
        show: bool = True,
    ):
        """Beeswarm summary plot over a sample of rows (plan §11.1)."""
        if sample is not None and len(X_raw) > sample:
            X_raw = X_raw.sample(n=sample, random_state=seed)
        X_fe = self.feature_engineer.transform(X_raw)
        sv = self.explainer.shap_values(X_fe)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        return shap.summary_plot(
            sv,
            X_fe,
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=show,
        )

    def bar_plot(
        self,
        X_raw: pd.DataFrame,
        sample: int | None = 10_000,
        seed: int = 42,
        show: bool = True,
    ):
        """Mean |SHAP| bar plot (plan §11.1 — most defensible importance)."""
        if sample is not None and len(X_raw) > sample:
            X_raw = X_raw.sample(n=sample, random_state=seed)
        X_fe = self.feature_engineer.transform(X_raw)
        sv = self.explainer.shap_values(X_fe)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        return shap.summary_plot(
            sv,
            X_fe,
            feature_names=self.feature_names,
            plot_type="bar",
            show=show,
        )

    def waterfall_for_row(
        self,
        X_raw: pd.DataFrame,
        row_idx: int = 0,
        max_display: int = 12,
        show: bool = True,
    ):
        """Waterfall plot for a single transaction (plan §11.2)."""
        X_fe = self.feature_engineer.transform(X_raw.iloc[[row_idx]])
        sv = self.explainer.shap_values(X_fe)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        ev = self.expected_value()
        explanation = shap.Explanation(
            values=np.asarray(sv)[0],
            base_values=ev,
            data=X_fe.iloc[0].to_numpy(),
            feature_names=self.feature_names,
        )
        return shap.plots.waterfall(explanation, max_display=max_display, show=show)


__all__ = ["FeatureContribution", "FraudExplainer"]
