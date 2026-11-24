import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="NIRF 2026 Predictor", layout="wide")

# --- LOAD DATA & MODELS ---
@st.cache_resource
def load_assets():
    df = pd.read_csv('NIRF_Combined_2017_2025.csv')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['TLR', 'RPC', 'GO', 'OI', 'Perception', 'Score', 'Rank'])
    
    # Load the pre-trained models
    score_model = joblib.load('nirf_score_model.pkl')
    rank_model = joblib.load('nirf_rank_model.pkl')
    
    return df, score_model, rank_model

df, score_model, rank_model = load_assets()

# --- CONSTANTS ---
metrics = ['TLR', 'RPC', 'GO', 'OI', 'Perception']
nirf_weights = {'TLR': 0.30, 'RPC': 0.20, 'GO': 0.30, 'OI': 0.10, 'Perception': 0.10}
improvement_actions = {
    'TLR': 'Increase faculty strength and PhD-qualified staff.',
    'RPC': 'Boost funded research and Scopus-indexed publications.',
    'GO': 'Improve placement outcomes and higher studies tracking.',
    'OI': 'Expand scholarships and diversity inclusion.',
    'Perception': 'Enhance branding and alumni engagement.'
}

# --- SIDEBAR ---
st.sidebar.header("Prediction Settings")
college_list = sorted(df['Name'].unique())
selected_college = st.sidebar.selectbox("Select Institution", college_list)

# --- MAIN UI ---
st.title("🎓 NIRF 2026 Rank Prediction System")
st.markdown(f"Predicting future performance for **{selected_college}** based on historical trends (2017-2025).")

# --- LOGIC ---
college_df = df[df['Name'] == selected_college].sort_values('Year')

if not college_df.empty:
    # 1. Forecast 2026 Parameter Scores using Linear Regression
    forecast_2026 = {}
    for m in metrics:
        reg = LinearRegression().fit(college_df[['Year']], college_df[m])
        pred = reg.predict([[2026]])[0]
        forecast_2026[m] = np.clip(pred, 0, 100)

    # 2. Predict Final Score and Rank
    input_df = pd.DataFrame([forecast_2026])
    pred_score = score_model.predict(input_df)[0]
    
    rank_input = pd.DataFrame({'Score': [pred_score], 'Year': [2026]})
    pred_rank = max(1, int(round(rank_model.predict(rank_input)[0])))

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted 2026 Score", f"{pred_score:.2f}")
    col2.metric("Predicted 2026 Rank", f"#{pred_rank}")
    col3.metric("Latest Rank (2025)", f"#{int(college_df['Rank'].iloc[-1])}")

    # --- VISUALIZATIONS ---
    st.subheader("Parameter Forecast & Improvement Priority")
    
    # Calculate Impact for the bar chart
    impact_scores = {m: (100 - forecast_2026[m]) * nirf_weights[m] for m in metrics}
    priority_metric = max(impact_scores, key=impact_scores.get)

    fig, ax = plt.subplots()
    ax.bar(impact_scores.keys(), impact_scores.values(), color='skyblue')
    ax.set_ylabel("Improvement Impact Score")
    st.pyplot(fig)

    # --- RECOMMENDATION BOX ---
    st.info(f"**Strategic Recommendation:** {improvement_actions[priority_metric]}")
    
    with st.expander("View Predicted Parameter Breakdown"):
        st.write(pd.DataFrame([forecast_2026], index=["2026 Prediction"]))

else:
    st.error("Data not found for this institution.")