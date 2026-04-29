"""Feature engineering as an sklearn transformer.

Implements plan §6: time features, amount features, and a scaling strategy
that depends on the model family. Wrapped as a single transformer so the
exact same transformations are applied at training time and at inference
inside the FastAPI service — the plan calls this out as the single most
important defence against train-serve skew (§6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

V_COLS = [f"V{i}" for i in range(1, 29)]


@dataclass
class FeatureConfig:
    """Toggle which engineered features are produced.

    Defaults match the recommended set for tree-based models. Set
    `include_amount_bucket=False` and add polynomial Amount features in
    a downstream step if you want a linear-model baseline.
    """

    add_hour_of_day: bool = True
    add_is_night: bool = True
    add_log_amount: bool = True
    add_amount_zero_flag: bool = True
    add_amount_bucket: bool = True
    n_amount_buckets: int = 10
    night_hours: tuple[int, int] = (0, 6)  # half-open [start, end)
    interaction_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless* feature engineer for the credit-card fraud dataset.

    *The only learned state is the Amount-bucket edges, fitted on TRAIN
    only (qcut quantile boundaries) and reused at inference time. All
    other features are pure functions of the input row, so a single
    transaction can be scored in constant time during API calls.

    Input columns expected: Time, V1..V28, Amount.
    Output columns: original V1..V28 plus the engineered features.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    # ---- sklearn API ----------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, ARG002
        self._validate_input_columns(X)
        cfg = self.config

        if cfg.add_amount_bucket:
            # Quantile edges from TRAIN only — never refit at inference.
            _, edges = pd.qcut(
                X["Amount"],
                q=cfg.n_amount_buckets,
                retbins=True,
                duplicates="drop",
            )
            # Force open intervals on both ends so unseen amounts at
            # inference time still bucket cleanly.
            edges = np.array(edges, dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.amount_bucket_edges_ = edges
        else:
            self.amount_bucket_edges_ = None

        # Record the output schema so downstream pipeline steps and the
        # API can validate feature names.
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = self._compute_output_columns()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_columns(X)
        if not hasattr(self, "amount_bucket_edges_"):
            raise RuntimeError("FeatureEngineer.transform called before fit")

        cfg = self.config
        out = X[V_COLS].copy()

        if cfg.add_hour_of_day:
            out["hour_of_day"] = ((X["Time"].to_numpy() // 3600) % 24).astype(int)

        if cfg.add_is_night:
            hours = (X["Time"].to_numpy() // 3600) % 24
            start, end = cfg.night_hours
            out["is_night"] = ((hours >= start) & (hours < end)).astype(int)

        # Always keep Time itself (drift signal across the 2-day window),
        # plus Amount itself for tree models that may want the raw value.
        out["seconds_since_start"] = X["Time"].to_numpy()
        out["amount"] = X["Amount"].to_numpy()

        if cfg.add_log_amount:
            out["log_amount"] = np.log1p(X["Amount"].to_numpy())

        if cfg.add_amount_zero_flag:
            out["amount_zero_flag"] = (X["Amount"].to_numpy() == 0).astype(int)

        if cfg.add_amount_bucket and self.amount_bucket_edges_ is not None:
            buckets = (
                np.digitize(X["Amount"].to_numpy(), self.amount_bucket_edges_) - 1
            )
            # Clip into [0, n_buckets-1] so unseen extremes never land on
            # a bucket index outside the trained range.
            n = cfg.n_amount_buckets
            buckets = np.clip(buckets, 0, n - 1)
            out["amount_bucket"] = buckets.astype(int)

        for a, b in cfg.interaction_pairs:
            if a not in V_COLS or b not in V_COLS:
                raise ValueError(
                    f"Interaction pair {(a, b)} must reference V1..V28 columns"
                )
            out[f"{a}_x_{b}"] = X[a].to_numpy() * X[b].to_numpy()

        # Reorder so the column ordering at fit time matches transform time.
        out = out[self.feature_names_out_]
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError(
                "get_feature_names_out called before fit; call fit first"
            )
        return np.array(self.feature_names_out_, dtype=object)

    # ---- helpers --------------------------------------------------------

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        required = {"Time", "Amount", *V_COLS}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

    def _compute_output_columns(self) -> list[str]:
        cfg = self.config
        cols: list[str] = list(V_COLS)
        if cfg.add_hour_of_day:
            cols.append("hour_of_day")
        if cfg.add_is_night:
            cols.append("is_night")
        cols.extend(["seconds_since_start", "amount"])
        if cfg.add_log_amount:
            cols.append("log_amount")
        if cfg.add_amount_zero_flag:
            cols.append("amount_zero_flag")
        if cfg.add_amount_bucket:
            cols.append("amount_bucket")
        for a, b in cfg.interaction_pairs:
            cols.append(f"{a}_x_{b}")
        return cols


__all__ = ["FeatureConfig", "FeatureEngineer", "V_COLS"]
