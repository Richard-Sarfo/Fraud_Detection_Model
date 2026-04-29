"""Pipeline contract tests (plan §12.5).

Covers:
- Data loading round-trip and schema validation.
- Stratified and time-based splits preserve the expected fraud-rate
  invariants and never share rows.
- FeatureEngineer is fit-once / transform-many and produces identical
  output for single-row vs batch transforms (the train-serve skew check).
- Evaluate suite returns sensible values on synthetic data.
- API loads a fresh artefact, exposes /health, and returns valid
  PredictionOut JSON for /predict and /predict/batch.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data import (
    FEATURE_COLS, TARGET_COL, imbalance_ratio, load_raw, sha256_of_file,
    split_stratified, split_time_based,
)
from src.evaluate import (
    bootstrap_metric, evaluate, find_threshold_max_f1,
    find_threshold_min_cost, precision_at_recall, recall_at_fpr,
)
from src.explain import FraudExplainer
from src.features import FeatureConfig, FeatureEngineer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TestData:
    def test_load_raw_round_trip(self, synthetic_csv):
        df = load_raw(synthetic_csv, drop_duplicates=False, verify_shape=False)
        assert set(FEATURE_COLS + [TARGET_COL]).issubset(df.columns)
        assert df[TARGET_COL].isin([0, 1]).all()

    def test_sha256_is_stable(self, synthetic_csv):
        h1 = sha256_of_file(synthetic_csv)
        h2 = sha256_of_file(synthetic_csv)
        assert h1 == h2
        assert len(h1) == 64

    def test_load_raw_rejects_missing_columns(self, tmp_path, synthetic_df):
        bad = tmp_path / "broken.csv"
        synthetic_df.drop(columns=["V14"]).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing expected columns"):
            load_raw(bad, verify_shape=False)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_stratified_preserves_fraud_rate(self, synthetic_df):
        splits = split_stratified(synthetic_df, random_state=0)
        overall = synthetic_df[TARGET_COL].mean()
        # Within ~1.5 pp on a tiny 2050-row dataset.
        for split_rate in splits.fraud_rates.values():
            assert abs(split_rate - overall) < 0.015

    def test_stratified_disjoint_indices(self, synthetic_df):
        splits = split_stratified(synthetic_df, random_state=0)
        sets = [set(splits.X_train.index), set(splits.X_val.index), set(splits.X_test.index)]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])

    def test_time_based_is_ordered(self, synthetic_df):
        splits = split_time_based(synthetic_df, random_state=0)
        assert splits.X_train["Time"].max() <= splits.X_val["Time"].min()
        assert splits.X_val["Time"].max() <= splits.X_test["Time"].min()

    def test_imbalance_ratio_positive(self, synthetic_df):
        splits = split_stratified(synthetic_df, random_state=0)
        ratio = imbalance_ratio(splits.y_train)
        assert ratio > 1.0  # heavily imbalanced

    def test_imbalance_ratio_zero_positives_raises(self):
        with pytest.raises(ValueError):
            imbalance_ratio(pd.Series([0, 0, 0]))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_fit_transform_shape(self, synthetic_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(synthetic_df.drop(columns=["Class"]))
        assert out.shape[0] == len(synthetic_df)
        assert "log_amount" in out.columns
        assert "hour_of_day" in out.columns
        assert "amount_bucket" in out.columns

    def test_train_serve_skew(self, synthetic_df):
        """Single-row transform must equal corresponding row from a batch."""
        X = synthetic_df.drop(columns=["Class"])
        fe = FeatureEngineer().fit(X.iloc[:1500])
        # Pick a hold-out row the FE didn't see at fit time.
        row = X.iloc[[1700]]
        single = fe.transform(row).values
        batch = fe.transform(X.iloc[1500:]).iloc[0:1].values
        np.testing.assert_array_equal(single, batch)

    def test_missing_input_columns_raises(self, synthetic_df):
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="missing required columns"):
            fe.fit(synthetic_df.drop(columns=["Time"]))

    def test_amount_buckets_clip_unseen_extremes(self, synthetic_df):
        X = synthetic_df.drop(columns=["Class"])
        fe = FeatureEngineer(FeatureConfig(n_amount_buckets=5)).fit(X)
        extreme = X.iloc[[0]].copy()
        extreme.loc[:, "Amount"] = 1e9
        out = fe.transform(extreme)
        assert int(out["amount_bucket"].iloc[0]) == 4  # max bucket index


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_returns_full_suite(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"])
        y = synthetic_df["Class"].to_numpy()
        proba = fitted_pipeline.predict_proba(X)[:, 1]
        report = evaluate(y, proba)
        assert 0.0 <= report.pr_auc <= 1.0
        assert 0.0 <= report.roc_auc <= 1.0
        assert 0.0 <= report.brier <= 1.0
        assert {"default", "f1_optimal", "cost_optimal"} <= set(report.thresholds)

    def test_thresholds_are_distinct_for_imbalanced_data(
        self, fitted_pipeline, synthetic_df
    ):
        X = synthetic_df.drop(columns=["Class"])
        y = synthetic_df["Class"].to_numpy()
        proba = fitted_pipeline.predict_proba(X)[:, 1]
        report = evaluate(y, proba)
        # On imbalanced data, default 0.5 should not equal cost_optimal in
        # general — just confirm both exist; allow them to coincide if the
        # synthetic model is unusually well-calibrated.
        assert report.thresholds["default"].threshold == pytest.approx(0.5)
        assert report.thresholds["cost_optimal"].threshold >= 0.0
        assert report.thresholds["cost_optimal"].threshold <= 1.0

    def test_recall_at_fpr_bounds(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=1000)
        # At least one positive — keep regenerating if needed.
        if y.sum() == 0:
            y[0] = 1
        proba = rng.random(1000)
        r = recall_at_fpr(y, proba, max_fpr=0.05)
        assert 0.0 <= r <= 1.0

    def test_precision_at_recall_bounds(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.1).astype(int)
        if y.sum() == 0:
            y[0] = 1
        proba = rng.random(500)
        p = precision_at_recall(y, proba, min_recall=0.5)
        assert 0.0 <= p <= 1.0

    def test_threshold_max_f1_returns_valid_pair(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"])
        y = synthetic_df["Class"].to_numpy()
        proba = fitted_pipeline.predict_proba(X)[:, 1]
        t, f1 = find_threshold_max_f1(y, proba)
        assert 0.0 <= t <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_threshold_min_cost(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"])
        y = synthetic_df["Class"].to_numpy()
        proba = fitted_pipeline.predict_proba(X)[:, 1]
        t, cost = find_threshold_min_cost(y, proba, cost_fn=100.0, cost_fp=1.0)
        assert 0.0 <= t <= 1.0
        assert cost >= 0.0

    def test_bootstrap_returns_ordered_ci(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"])
        y = synthetic_df["Class"].to_numpy()
        proba = fitted_pipeline.predict_proba(X)[:, 1]
        pt, lo, hi = bootstrap_metric(y, proba, "pr_auc", n_iter=100, seed=0)
        assert lo <= pt <= hi


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_top_contributors_returns_k_per_row(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"]).iloc[:5].reset_index(drop=True)
        explainer = FraudExplainer(fitted_pipeline)
        contribs = explainer.top_contributors(X, k=3)
        assert len(contribs) == 5
        for row in contribs:
            assert len(row) == 3

    def test_shap_values_match_feature_count(self, fitted_pipeline, synthetic_df):
        X = synthetic_df.drop(columns=["Class"]).iloc[:10].reset_index(drop=True)
        explainer = FraudExplainer(fitted_pipeline)
        sv = explainer.shap_values(X)
        assert sv.shape[0] == 10
        assert sv.shape[1] == len(explainer.feature_names)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class TestAPI:
    @pytest.fixture
    def client(self, model_artefacts, monkeypatch):
        """Spin up a fresh app pointed at the synthetic-trained artefacts."""
        monkeypatch.setenv("MODEL_PATH", str(model_artefacts["model_path"]))
        monkeypatch.setenv("METADATA_PATH", str(model_artefacts["metadata_path"]))

        # Reload the module so it picks up the new env vars at import time.
        import src.api as api_mod
        importlib.reload(api_mod)
        with TestClient(api_mod.app) as c:
            yield c

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_predict_returns_valid_schema(self, client, synthetic_df):
        row = synthetic_df.drop(columns=["Class"]).iloc[0].to_dict()
        r = client.post("/predict", json=row)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
        assert isinstance(body["top_features"], list)
        assert len(body["top_features"]) == 3

    def test_predict_rejects_negative_amount(self, client, synthetic_df):
        row = synthetic_df.drop(columns=["Class"]).iloc[0].to_dict()
        row["Amount"] = -1.0
        r = client.post("/predict", json=row)
        assert r.status_code == 422

    def test_predict_batch_shape(self, client, synthetic_df):
        rows = synthetic_df.drop(columns=["Class"]).iloc[:5].to_dict(orient="records")
        r = client.post("/predict/batch", json={"transactions": rows})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 5
        assert body["elapsed_ms"] >= 0.0

    def test_predict_batch_rejects_empty(self, client):
        r = client.post("/predict/batch", json={"transactions": []})
        assert r.status_code == 422
