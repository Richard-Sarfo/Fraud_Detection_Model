# Fraud Detection Model

End-to-end credit-card fraud detection on the Kaggle MLG-ULB dataset.
Implements EDA → feature engineering → imbalance experiments → Optuna
tuning → single-shot test evaluation → SHAP explainability → FastAPI
serving, all inside Docker.

> **Plan:** see `fraud_detection_ml_model_plan.pdf` for the full §1–§15
> design. The PR descriptions cite specific sections.

## Quickstart (Docker only)

You only need Docker Desktop on the host — nothing else.

```bash
# 1. Place the Kaggle CSV in Data/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
ls Data/creditcard.csv     # ~144 MB

# 2. Build the image once
docker compose build

# 3. Run notebooks (JupyterLab on http://localhost:8888)
docker compose up jupyter

# 4. Train the champion (writes models/champion.pkl + metadata.json)
docker compose --profile train up train

# 5. Serve predictions (http://localhost:8000)
docker compose up api

# 6. Run the test suite
docker compose --profile test up test
```

`docker compose up` starts JupyterLab and the API together.

## Project layout

```
fraud-detection/
├── Data/creditcard.csv      # 144 MB, git-ignored
├── data/processed/          # train/val/test parquet (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb              # plan §5
│   ├── 02_feature_engineering.ipynb  # plan §6, §7
│   ├── 03_modeling.ipynb         # plan §8, §9, §10
│   └── 04_explainability.ipynb   # plan §11
├── src/
│   ├── data.py        # load + split + dataset hash (§2, §5.1, §7, §13)
│   ├── features.py    # FeatureEngineer transformer (§6)
│   ├── evaluate.py    # PR-AUC, thresholds, bootstrap CIs (§10)
│   ├── train.py       # Optuna pipeline; single-shot test (§8, §9, §13)
│   ├── explain.py     # SHAP TreeExplainer wrappers (§11)
│   └── api.py         # FastAPI service (§12)
├── models/                  # champion.pkl + metadata.json (git-ignored)
├── tests/                   # contract tests (§12.5)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt         # exact pins for reproducibility (§13)
```

## API endpoints (plan §12.2)

After `docker compose up api`:

- `GET  /health` — model load status, version, deployed threshold
- `POST /predict` — score one transaction; returns probability,
  decision (`APPROVE` / `REVIEW` / `DECLINE`), threshold used,
  model version, top-3 SHAP contributors
- `POST /predict/batch` — array form, with elapsed-ms in the response
- OpenAPI docs at `http://localhost:8000/docs`

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "Time": 0,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
  "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
  "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
  "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}
JSON
```

## Training options

```bash
# Inside the train container; or run via `docker compose run --rm train ...`
python -m src.train --trials 50 --folds 5 --seed 42 \
                    --cost-fn 100 --cost-fp 1
```

Key flags:
- `--trials`: Optuna budget (50–100 recommended for the portfolio run; the
  plan §9.2 calls out diminishing returns past 100 on a 285k-row dataset).
- `--cost-fn / --cost-fp`: dollar costs per false negative / false positive.
  These set the cost-optimal threshold written to `metadata.json` and used
  by the API for the `decision` field. Defaults model a typical fraud
  business ($100/FN, $1/FP).

## Success criteria (plan §1.2)

| Metric                    | Target          |
|---------------------------|-----------------|
| PR-AUC (test set)         | ≥ 0.80          |
| Recall at 1 % FPR         | ≥ 0.80          |
| F1 at chosen threshold    | ≥ 0.85          |
| Inference latency (local) | ≤ 50 ms p99     |

Reported by `src.train.run` and persisted to `models/metadata.json`.

## Reproducibility (plan §13)

- All random seeds fixed via `--seed` (default 42).
- `models/metadata.json` records dataset SHA-256, library versions,
  best params, CV PR-AUC mean/std, full test metrics, chosen threshold.
- `requirements.txt` pins exact versions; pickled models break across
  versions.

## Risks & limitations (plan §15)

The PCA features do not generalize beyond this dataset — this model is a
portfolio artefact, not a transferable fraud detector.
