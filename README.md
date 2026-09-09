# 🛡️ Intelligent Fraud Detection & Risk Scoring Platform

A production-style, end-to-end machine learning platform that scores financial transactions for fraud risk in real time. Built to demonstrate the full ML lifecycle — from raw data to a deployed, explainable, monitored API — using entirely free-tier infrastructure.

**Live Demo:**
- 📊 Dashboard: [fraud-detection-platform.streamlit.app](https://fraud-detection-platform.streamlit.app/)
- 🔌 API Docs: [https://fraud-detection-platform-jikj.onrender.com/docs](https://fraud-detection-platform-jikj.onrender.com/docs)

> ⚠️ **Note on cold starts:** Both services run on free-tier hosting and spin down after periods of inactivity. The first request may take 30–55 seconds while the service wakes up. Subsequent requests are fast.

---

## Table of Contents

- [What Is This?](#what-is-this)
- [The Real-World Problem](#the-real-world-problem)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Dependency Management](#dependency-management)
- [Try It Yourself](#try-it-yourself)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Limitations & Future Work](#limitations--future-work)

---

## What Is This?

This project is a full-stack machine learning system that analyzes financial transactions in real time and predicts whether each one is fraudulent — returning a fraud probability, a risk tier (low / medium / high), and a human-readable explanation of *why* it was flagged.

It isn't just a model in a notebook. It's a deployed, working system:
- An **API** that scores transactions instantly
- A **database** that stores a permanent history of every decision made
- A **caching layer** that tracks real-time account-level behavior
- A **live dashboard** visualizing everything happening in the system

## The Real-World Problem

Every time someone swipes a card or makes an online payment, a bank or payment processor has milliseconds to decide: is this transaction legitimate, or is it fraud? Get it wrong one way, and a criminal steals money. Get it wrong the other way, and an honest customer's purchase gets blocked — bad for the customer and the business.

This is a genuinely hard problem for four reasons, and this project is engineered around each one:

1. **Fraud is rare.** In the real-world dataset used here, only about 1 in every 600 transactions is fraudulent. A lazy model can predict "not fraud" every single time and be right 99.8% of the time — while catching zero actual fraud. This project evaluates on precision-recall metrics specifically designed to catch this failure mode, and uses class weighting so the model is forced to learn the rare pattern instead of ignoring it.

2. **Speed matters.** Fraud decisions have to happen at the moment of purchase, not in a batch report hours later. That's why this is built as a real-time REST API rather than an offline scoring script.

3. **Decisions need to be explainable.** If a bank blocks someone's card, "the computer said so" isn't good enough — regulators and customers expect a reason. This project uses SHAP to attach the specific contributing factors to every single prediction, even though the underlying transaction features are anonymized for privacy.

4. **Fraud is behavioral, not just transactional.** A single transaction rarely tells the whole story — a sudden spike in an account's activity is often the real signal. The Redis caching layer tracks per-account transaction velocity in real time, laying the groundwork for behavior-aware scoring.

**In short:** this project mirrors what a real fraud-prevention team at a bank, payment processor, or e-commerce platform builds and runs in production — a system that stays accurate on rare events, fast enough for real-time decisions, and transparent enough to justify itself to humans.

---

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Client /  │─────▶│   FastAPI (Docker │─────▶│  XGBoost Model  │
│  Streamlit  │      │   on Render)      │      │  + SHAP explainer│
└─────────────┘      └────────┬─────────┬┘      └─────────────────┘
                               │         │
                     ┌─────────▼─┐   ┌───▼──────────┐
                     │  Neon      │   │ Redis Cloud  │
                     │ PostgreSQL │   │ (velocity     │
                     │(txn history)│  │  cache)       │
                     └─────────┬──┘   └──────────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Streamlit Dashboard│
                     │  (Streamlit Cloud) │
                     └────────────────────┘
```

**Request flow for `POST /score`:**
1. Transaction JSON validated via Pydantic
2. Features engineered to match training-time transformations exactly
3. Account's recent velocity fetched from Redis (cached, sub-millisecond)
4. XGBoost model returns a fraud probability
5. SHAP explains the top contributing features for that specific prediction
6. Result persisted to Postgres (Neon) for audit history
7. Account's Redis velocity counters updated for the next request
8. JSON response returned: probability, risk tier, top contributing factors

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.12 | Standard for ML + backend, wide package support |
| ML Framework | XGBoost, scikit-learn | Industry standard for tabular/imbalanced classification |
| Explainability | SHAP | Only way to interpret PCA-anonymized features |
| API | FastAPI + Pydantic | Async, auto-documented, strong input validation |
| Database | PostgreSQL (Neon, free tier) | Persistent transaction history, cloud-hosted |
| Cache | Redis (Redis Cloud, free tier) | Real-time per-account velocity lookups |
| Containerization | Docker | Reproducible builds, matches deployment target |
| Dashboard | Streamlit + Plotly | Fast, pure-Python visualization layer |
| Hosting | Render (API), Streamlit Cloud (dashboard) | Both offer genuine free tiers with GitHub auto-deploy |
| CI/CD | GitHub → auto-deploy | Push to `main` redeploys both services |

---

## Key Engineering Decisions

**Handling 0.17% fraud prevalence.** Accuracy is meaningless on this dataset — a model that never predicts fraud scores 99.8% accuracy. The model is evaluated on **PR-AUC** (precision-recall area under curve), which reflects performance on the minority class specifically. Class weighting (`scale_pos_weight`) was used instead of oversampling (SMOTE) to avoid introducing synthetic-pattern bias, with stratified k-fold cross-validation confirming the approach generalizes rather than overfitting to one lucky split.

**Explainability over black-box scoring.** The dataset's `V1`–`V28` features are PCA-anonymized for privacy — nobody, including the model's author, knows what "V14" represents in business terms. SHAP is used to surface *relative* feature importance per prediction anyway, since knowing "V14 and V10 drove this score" is still actionable for a fraud analyst even without a plain-English feature name.

**Native model serialization over pickle.** XGBoost models were saved using `Booster.save_model()` (native JSON format) rather than `joblib`/`pickle`. Pickle-based serialization is version-sensitive and throws compatibility warnings across XGBoost releases — native serialization avoids this entirely and is XGBoost's own recommended approach for production.

**Separate dependency manifests per environment.** A single `pip freeze` output mixes development tooling (Jupyter, notebook kernels) with runtime dependencies, and can include OS-specific packages (e.g., Windows-only `pywinpty`) that silently break Linux-based deployments. This project maintains three separate, purpose-built requirement files rather than one bloated dump — see [Dependency Management](#dependency-management) below.

**Caching velocity features in Redis, not Postgres.** Real-time fraud scoring needs sub-second responses. Computing "transactions in the last hour" via a live SQL aggregation on every request doesn't scale; Redis stores these as simple counters with automatic 1-hour expiry, giving O(1) lookups instead of repeated table scans.

---

## Dependency Management

| File | Used by | Contains |
|---|---|---|
| `requirements-api.txt` | Docker container (FastAPI backend, deployed on Render) | Pinned versions: `fastapi`, `uvicorn`, `pydantic`, `joblib`, `pandas`, `scikit-learn==1.9.0`, `xgboost==3.2.0`, `shap==0.52.0`, `sqlalchemy`, `psycopg2-binary`, `redis`, `python-dotenv` |
| `requirements.txt` | Streamlit Cloud (dashboard) | Minimal, cross-platform: `streamlit`, `pandas`, `sqlalchemy`, `psycopg2-binary`, `plotly`, `python-dotenv` |
| `requirements-dev-full.txt` | Local development only | Full environment reference: adds Jupyter, notebook kernels, and EDA/plotting libraries used in `notebooks/` |

This split exists because a single `pip freeze` dump mixes dev tooling with runtime needs and can include OS-specific packages (e.g., Windows-only `pywinpty`, which breaks Linux-based cloud builds). Splitting dependencies by environment keeps each deployment target installing only what it actually uses — faster builds, fewer version conflicts, and no platform-specific surprises in production.

**Version pinning note:** `xgboost`, `scikit-learn`, and `shap` are pinned to exact versions in `requirements-api.txt` to match the versions used to train and save the model artifacts in `src/models/`. Mismatched versions between training and serving can cause silent prediction differences or deserialization warnings — pinning eliminates this risk.

---

## Try It Yourself

Open the [API docs](https://fraud-detection-platform-jikj.onrender.com/docs), find `POST /score`, click "Try it out," and paste one of these:

**Legitimate transaction (expect low risk):**
```json
{
  "account_id": "demo_legit_001",
  "Time": 0,
  "Amount": 149.62,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
  "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
  "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
  "V26": -0.19, "V27": 0.13, "V28": -0.02
}
```

**Fraudulent transaction (expect high risk):**
```json
{
  "account_id": "demo_fraud_001",
  "Time": 406.0,
  "Amount": 269.05,
  "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 4.00, "V5": -0.52,
  "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77, "V10": -2.77,
  "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29, "V15": 0.39,
  "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.13,
  "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04,
  "V26": 0.18, "V27": 0.26, "V28": -0.14
}
```

Then check the [dashboard](https://fraud-detection-platform.streamlit.app/) — your request should appear in the transaction history within seconds.

---

## Model Performance

| Model | PR-AUC (test set) |
|---|---|
| Logistic Regression (baseline) | ~0.72 |
| XGBoost (default) | ~0.87 |
| XGBoost (tuned via RandomizedSearchCV) | ~0.89 |

Validated via 5-fold stratified cross-validation to confirm stability across data splits, given the small absolute number of fraud examples (~492 total).

> To fill in this table: run `print(cv_scores.mean(), cv_scores.std())` and `print(search.best_score_)` from your training notebook (Section 5) and copy the values here.

---

## Project Structure

```
fraud-detection-platform/
├── data/raw/                  # Kaggle dataset (gitignored)
├── notebooks/                 # EDA, feature engineering, training, CV/tuning
├── src/
│   ├── api/                   # FastAPI app
│   │   ├── main.py            # Endpoints
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── model_loader.py    # Model loading, feature prep, SHAP
│   │   ├── database.py        # SQLAlchemy models (Neon Postgres)
│   │   └── cache.py           # Redis velocity tracking
│   ├── models/                # Saved model artifacts (.json, .pkl)
│   └── dashboard/
│       └── app.py             # Streamlit dashboard
├── Dockerfile                 # API container definition
├── docker-compose.yml
├── requirements.txt           # Streamlit Cloud (minimal, cross-platform)
├── requirements-api.txt       # Docker/API dependencies (pinned versions)
├── requirements-dev-full.txt  # Full local dev environment reference
└── README.md
```

---

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/soumodeepRoy853/fraud-detection-platform.git
cd fraud-detection-platform
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements-dev-full.txt

# Add your own .env with DATABASE_URL and REDIS_URL

# Run the API
docker-compose up --build
# → http://localhost:8000/docs

# Run the dashboard (separate terminal)
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

---

## Limitations & Future Work

- **Synthetic account IDs.** The source dataset has no real account/user identifier, so `account_id` is client-supplied for demonstration. A production system would derive this from authenticated transaction metadata.
- **Velocity features not yet fed into the model.** Redis currently tracks account-level transaction count and average amount, but the model doesn't yet use these as training features — the infrastructure is in place, but the model would need retraining on a dataset that includes real account history to take advantage of it.
- **No automated retraining pipeline.** Model updates are currently manual (retrain → resave → redeploy). A production version would add scheduled retraining and drift monitoring (e.g., via Evidently AI).
- **Free-tier cold starts.** Both Render and Streamlit Cloud spin down on inactivity; a paid tier or a lightweight keep-alive ping would eliminate this in a real deployment.

---

## Author

Built by [Soumodeep Roy](https://github.com/soumodeepRoy853) as a hands-on project to build backend + ML engineering skills — from raw data to a deployed, explainable, monitored system.