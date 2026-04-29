"""FastAPI app exposing /health, /predict, and /predict/batch endpoints.

Implements plan §12 (saving and serving the model locally). Designed
for the docker-compose `api` service: mounts ./models read-only,
loads ./models/champion.pkl and ./models/metadata.json at startup,
serves Pydantic-validated predictions with SHAP top-features and a
deployed-threshold-based decision.

Run inside Docker:
    docker compose up api

Or directly:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

import src
from src.explain import FraudExplainer

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/champion.pkl"))
DEFAULT_METADATA_PATH = Path(
    os.environ.get("METADATA_PATH", "models/metadata.json")
)


# ---------------------------------------------------------------------------
# Pydantic schemas (plan §12.3)
# ---------------------------------------------------------------------------


class TransactionIn(BaseModel):
    """One credit-card transaction row, matching the Kaggle schema."""

    Time: float = Field(ge=0, description="Seconds since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float
    V14: float; V15: float; V16: float; V17: float; V18: float; V19: float
    V20: float; V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(ge=0)


class FeatureContributionOut(BaseModel):
    name: str
    value: float
    shap_value: float


class PredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    threshold_used: float
    model_version: str
    top_features: list[FeatureContributionOut]


class BatchIn(BaseModel):
    transactions: list[TransactionIn]


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    elapsed_ms: float


class HealthOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "error"]
    model_loaded: bool
    model_version: str | None
    threshold: float | None
    library_version: str = src.__version__


# ---------------------------------------------------------------------------
# State & lifespan
# ---------------------------------------------------------------------------


class AppState:
    pipeline = None
    explainer: FraudExplainer | None = None
    threshold: float = 0.5
    review_threshold: float | None = None
    metadata: dict = {}


def _decide(prob: float, threshold: float, review_threshold: float | None) -> str:
    """Three-bucket decision: DECLINE / REVIEW / APPROVE.

    REVIEW is used when the score is in the grey zone (>= review_threshold
    but below the deployed cost-optimal threshold). If review_threshold
    is None, fall back to a binary APPROVE/DECLINE.
    """
    if prob >= threshold:
        return "DECLINE"
    if review_threshold is not None and prob >= review_threshold:
        return "REVIEW"
    return "APPROVE"


def _load_artefacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `docker compose --profile train up train` first."
        )
    AppState.pipeline = joblib.load(model_path)
    AppState.explainer = FraudExplainer(AppState.pipeline)

    if metadata_path.exists():
        AppState.metadata = json.loads(metadata_path.read_text())
        deployment = AppState.metadata.get("deployment", {})
        AppState.threshold = float(deployment.get("chosen_threshold", 0.5))
        # Halfway between cost_optimal and 1.0 is too aggressive; instead
        # use cost_optimal / 2 as the REVIEW lower bound — flags marginal
        # cases without overwhelming the queue. Override via env if needed.
        rt_env = os.environ.get("REVIEW_THRESHOLD")
        AppState.review_threshold = (
            float(rt_env) if rt_env is not None else AppState.threshold / 2.0
        )
    else:
        logger.warning("metadata.json not found at %s; using threshold=0.5", metadata_path)

    logger.info(
        "Loaded model from %s (threshold=%.4f, review=%s)",
        model_path,
        AppState.threshold,
        AppState.review_threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    try:
        _load_artefacts()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        # Keep the app up so /health can report the failure rather than
        # making the container crash-loop.
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Credit-card fraud scoring with SHAP explanations.",
    version=src.__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    if AppState.pipeline is None:
        return HealthOut(
            status="error",
            model_loaded=False,
            model_version=None,
            threshold=None,
        )
    return HealthOut(
        status="ok",
        model_loaded=True,
        model_version=AppState.metadata.get("model_version", src.__version__),
        threshold=AppState.threshold,
    )


def _score_dataframe(df: pd.DataFrame) -> list[PredictionOut]:
    if AppState.pipeline is None or AppState.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded; check /health and the train profile.",
        )
    probs = AppState.pipeline.predict_proba(df)[:, 1]
    contributors = AppState.explainer.top_contributors(df, k=3)

    out: list[PredictionOut] = []
    model_version = AppState.metadata.get("model_version", src.__version__)
    for prob, top in zip(probs, contributors, strict=True):
        out.append(
            PredictionOut(
                fraud_probability=float(prob),
                decision=_decide(
                    float(prob), AppState.threshold, AppState.review_threshold
                ),
                threshold_used=AppState.threshold,
                model_version=model_version,
                top_features=[
                    FeatureContributionOut(**c.as_dict()) for c in top
                ],
            )
        )
    return out


@app.post("/predict", response_model=PredictionOut)
def predict(tx: TransactionIn) -> PredictionOut:
    df = pd.DataFrame([tx.model_dump()])
    return _score_dataframe(df)[0]


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn) -> BatchOut:
    if not batch.transactions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="transactions list cannot be empty",
        )
    t0 = time.perf_counter()
    df = pd.DataFrame([tx.model_dump() for tx in batch.transactions])
    preds = _score_dataframe(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return BatchOut(predictions=preds, elapsed_ms=elapsed_ms)


__all__ = ["app"]
