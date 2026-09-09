import traceback
from sqlalchemy.orm import Session
from src.api.database import get_db, TransactionLog
from fastapi import FastAPI, HTTPException, Depends
from src.api.schemas import Transaction, ScoreResponse
from src.api.model_loader import model, prepare_features, get_risk_tier, get_shap_explanation
from src.api.cache import get_account_velocity, update_account_velocity

app = FastAPI(title="Fraud Detection & Risk Scoring API")

@app.get('/')
def health_check():
    return {'status': 'API is running'}


@app.post('/score', response_model=ScoreResponse)
def score_transaction(transaction: Transaction, db: Session = Depends(get_db)):
    try:
        # Fetch cache velocity before scoring
        velocity = get_account_velocity(transaction.account_id)

        features = prepare_features(transaction.dict())
        probability = model.predict_proba(features)[0][1]
        tier = get_risk_tier(probability)
        top_factors = get_shap_explanation(features)

        # Update cache velocity after scoring
        update_account_velocity(transaction.account_id, transaction.Amount)

        # Save to DB
        log_entry = TransactionLog(
            amount=transaction.Amount,
            fraud_probability=round(float(probability), 4),
            risk_tier=tier
        )
        db.add(log_entry)
        db.commit()

        return ScoreResponse(
            fraud_probability = round(float(probability), 4),
            risk_tier = tier,
            top_factors = top_factors
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))