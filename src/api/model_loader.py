import shap
import joblib
import pandas as pd
from xgboost import XGBClassifier

model = XGBClassifier()

model.load_model("src/models/fraud_model.json")
scaler = joblib.load("src/models/amount_scaler.pkl")

def prepare_features(transaction: dict) -> pd.DataFrame:
    df = pd.DataFrame([transaction])

    #Replicate Section 3 feature engineering exactly
    df['Amount_scaled'] = scaler.transform(df[['Amount']])
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop(['Time', 'Amount'], axis=1)

    # Ensure column order matches training data
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Hour']
    df = df[expected_cols]

    return df


def get_risk_tier(probability: float) -> str:
    if probability < 0.3:
        return 'low'
    elif probability < 0.7:
        return 'medium'
    else:
        return 'high'


explainer = shap.TreeExplainer(model)

def get_shap_explanation(features_df, top_n=3):
    shap_values = explainer.shap_values(features_df)

    # Get shap values for 2D array in single row
    row_shap = shap_values[0]
    feature_names = features_df.columns.tolist()

    #Pair feature names with their corresponding shap contributions
    contributions = list(zip(feature_names, row_shap))

    #Sort by absolute contribution and get top_n
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:top_n]

    return [
        {'feature': name, 'impact': round(float(value), 4)}
        for name, value in top_contributions
    ]