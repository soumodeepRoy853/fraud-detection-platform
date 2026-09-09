import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import plotly.express as px

load_dotenv()

st.set_page_config(page_title='Fraud Detection Dashboard', layout='wide')

DATABASE_URL = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

@st.cache_data(ttl=30)
def load_data():
    query = 'SELECT * FROM transaction_logs ORDER BY score_at DESC'
    return pd.read_sql(query, engine)

df = load_data()

st.title('Fraud Detection & Risk Scoring Dashboard')

# Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric('Total Transactions Scored', len(df))
col2.metric('High Risk Flagged', len(df[df['risk_tier'] == 'high']))
fraud_rate = round((len(df[df['risk_tier'] == 'high']) / len(df)) * 100, 2) if len(df) > 0 else 0
col3.metric('Fraud Rate', f'{fraud_rate}%')


# Risk tier distribution
st.subheader("Risk Tier Distribution")
tier_counts = df['risk_tier'].value_counts().reset_index()
tier_counts.columns = ['risk_tier', 'count']
fig = px.bar(tier_counts, x='risk_tier', y='count', color='risk_tier',
            color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'})
st.plotly_chart(fig, use_container_width=True)

# Fraud probability over time 
st.subheader("Fraud Probability Over Time")
fig2 = px.scatter(df, x='score_at', y='fraud_probability', color='risk_tier',
                  color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'})
st.plotly_chart(fig2, use_container_width=True)

# Recent transactions table
st.subheader("Recent Transactions")
st.dataframe(df.head(50), use_container_width=True)