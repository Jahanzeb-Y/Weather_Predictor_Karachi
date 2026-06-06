import os
import requests
import pandas as pd
import streamlit as st
import hopsworks
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
os.environ["HOPSWORKS_API_KEY"] = "nqwhi0tLZlZzJQpq.jQ0OUyguCdCUbb11UoH4HA8qHmJmXyna27JEPJwJcszSet2W5GNRRopC7WpQZGSz"

st.set_page_config(page_title="Karachi 3-Day AQI Predictor", layout="wide")

st.markdown("# 🇵🇰 Karachi Serverless 3-Day AQI Forecast Engine")
st.markdown("This dashboard pulls live weather metrics, passes them into your trained AI model via Hopsworks, and predicts the air quality index 3 days into the future.")
st.markdown("---")

# 1. Fetch Live, Real-Time Weather Conditions from Open-Meteo
st.subheader("📡 Live Ambient Environmental Tracking")

live_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone"
response = requests.get(live_url).json()['hourly']
live_df = pd.DataFrame(response)

latest_row_idx = -1
cur_pm25 = live_df['pm2_5'].iloc[latest_row_idx]
cur_pm10 = live_df['pm10'].iloc[latest_row_idx]
cur_no2 = live_df['nitrogen_dioxide'].iloc[latest_row_idx]
cur_so2 = live_df['sulphur_dioxide'].iloc[latest_row_idx]
cur_ozone = live_df['ozone'].iloc[latest_row_idx]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Live $PM_{2.5}$", f"{cur_pm25} µg/m³")
col2.metric("Live $PM_{10}$", f"{cur_pm10} µg/m³")
col3.metric("Live $NO_2$", f"{cur_no2} µg/m³")
col4.metric("Live $SO_2$", f"{cur_so2} µg/m³")
col5.metric("Live Ozone", f"{cur_ozone} µg/m³")

st.markdown("---")

# 2. Download Model and Make Predictions
st.subheader("🔮 3-Day Lookahead Prediction")

@st.cache_resource
def download_ai_model():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project="Karachi_Weather_Forecast"
    )
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_prediction_model", version=2) # Using version 2 (with the updated metadata)
    model_dir = model_meta.download()
    loaded_model = joblib.load(f"{model_dir}/aqi_model.pkl")
    return loaded_model

try:
    with st.spinner("Downloading trained AI model parameters from Hopsworks Cloud Registry..."):
        model = download_ai_model()
    
    # 3. Engineer our runtime input payload to match what the model learned during training
    now = datetime.now()
    inference_payload = pd.DataFrame([{
        'pm2_5': cur_pm25,
        'pm10': cur_pm10,
        'nitrogen_dioxide': cur_no2,
        'sulphur_dioxide': cur_so2,
        'ozone': cur_ozone,
        'hour': now.hour,
        'day_of_week': now.weekday(),
        'month': now.month,
        'pm2_5_roll_6h': live_df['pm2_5'].iloc[-6:].mean(),
        'pm2_5_roll_24h': live_df['pm2_5'].iloc[-24:].mean(),
        'pm2_5_change_rate': float((live_df['pm2_5'].iloc[-1] - live_df['pm2_5'].iloc[-4]) / (live_df['pm2_5'].iloc[-4] + 1e-5))
    }])
    
    # Run the model!
    predicted_pm25 = model.predict(inference_payload)[0]
    
    # Display predictions alongside health risk advisory warnings
    st.markdown(f"### Predicted $PM_{2.5}$ concentration in 72 hours: **{predicted_pm25:.2f} µg/m³**")
    
    if predicted_pm25 < 35.0:
        st.success("🟢 **Air Quality Level: Good / Moderate** — Atmospheric particulates are predicted to remain within baseline security thresholds.")
    elif 35.0 <= predicted_pm25 < 150.0:
        st.warning("⚠️ **Air Quality Level: Unhealthy for Sensitive Groups** — High particulate concentrations predicted. People with respiratory sensitivities should take precautions.")
    else:
        st.error("🚨 **CRITICAL HEALTH HAZARD ALERT** — Heavily elevated atmospheric particulate levels expected. Limit outdoor exposures and close exterior ventilation access.")

    st.markdown("---")
    
    # NEW FEATURE: 📉 Advanced Analytics & Model Interpretability (SHAP)
    st.subheader("📊 Advanced Analytics: Feature Importance (SHAP)")
    st.markdown("This section calculates which weather features and indicators matter most to the AI model's forecast computations.")
    
    with st.spinner("Calculating SHAP mathematical dependency maps..."):
        # Create tree explainer tool for our Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(inference_payload)
        
        # Build a clean matplotlib bar plot
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # We look at index 0 because we're evaluating a single input row (current live data)
        # If it's a multi-output or tree array wrapper, grab the standard matrix list
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[0])
        else:
            vals = np.abs(shap_values)
            
        importance_df = pd.DataFrame({
            'Feature': inference_payload.columns,
            'SHAP Importance': vals[0]
        }).sort_values(by='SHAP Importance', ascending=True)
        
        # Plot horizontal bars
        ax.barh(importance_df['Feature'], importance_df['SHAP Importance'], color='#4682B4')
        ax.set_xlabel('Absolute Impact Score on 3-Day Forecast Prediction')
        plt.tight_layout()
        
        # Render directly inside your Streamlit Dashboard layout
        st.pyplot(fig)

except Exception as e:
    st.error(f"Failed to compile dashboard parameters or SHAP charts: {e}")