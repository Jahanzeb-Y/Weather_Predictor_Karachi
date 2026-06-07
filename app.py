import os
import requests
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pymongo import MongoClient

# --- CONFIGURATION & MONGO URI SELECTION ---
MONGO_URI = None

# 1. First, check if we are on the Streamlit Cloud server using secrets
try:
    if "MONGO_URI" in st.secrets:
        MONGO_URI = st.secrets["MONGO_URI"]
except Exception:
    # If st.secrets throws an error because the file doesn't exist locally, pass safely
    pass

# 2. If we are running locally, fall back to environment variables or hardcoded link string
if not MONGO_URI:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/?appName=KarachiAQIFeatures")

st.set_page_config(page_title="Karachi 3-Day AQI Predictor", layout="wide")

st.markdown("# 🇵🇰 Karachi Serverless 3-Day AQI Forecast Engine")
st.markdown("This dashboard pulls live weather metrics, passes them into your locally stored trained AI model, and predicts the air quality index 3 days into the future.")
st.markdown("---")

# --- MONGODB CONNECTION LAYER ---
@st.cache_resource
def get_mongo_collection():
    """Establishes a cached connection to the MongoDB Atlas cluster."""
    client = MongoClient(MONGO_URI)
    # Access your specific database and collection names
    db = client["Karachi_Weather_Forecast"]  # Replace with your actual DB name if different
    collection = db["karachi_aqi_features"]  # Replace with your actual collection name if different
    return collection

# 1. Fetch Live, Real-Time Weather Conditions from Open-Meteo
st.subheader("📡 Live Ambient Environmental Tracking")

live_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone"

try:
    response = requests.get(live_url).json()['hourly']
    api_map = {
        'pm2_5': response['pm2_5'],
        'pm10': response['pm10'],
        'nitrogen_dioxide': response['nitrogen_dioxide'],
        'sulphur_dioxide': response['sulphur_dioxide'],
        'ozone': response['ozone']
    }
    live_df = pd.DataFrame(api_map)
    live_df = live_df.dropna().reset_index(drop=True)

    # Grab the absolute latest complete record from the API
    cur_pm25 = float(live_df['pm2_5'].iloc[-1])
    cur_pm10 = float(live_df['pm10'].iloc[-1])
    cur_no2 = float(live_df['nitrogen_dioxide'].iloc[-1])
    cur_so2 = float(live_df['sulphur_dioxide'].iloc[-1])
    cur_ozone = float(live_df['ozone'].iloc[-1])

    # Display real-time data in UI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Live $PM_{2.5}$", f"{cur_pm25:.1f} µg/m³")
    col2.metric("Live $PM_{10}$", f"{cur_pm10:.1f} µg/m³")
    col3.metric("Live $NO_2$", f"{cur_no2} µg/m³")
    col4.metric("Live $SO_2$", f"{cur_so2} µg/m³")
    col5.metric("Live Ozone", f"{cur_ozone} µg/m³")

except Exception as api_err:
    st.error(f"Error accessing or parsing Open-Meteo API payload: {api_err}")
    cur_pm25, cur_pm10, cur_no2, cur_so2, cur_ozone = 30.0, 45.0, 15.0, 5.0, 20.0

st.markdown("---")

# 2. Fetch Historical Database Context for Rolling Calculations
with st.spinner("Querying historical baseline records from MongoDB Atlas..."):
    try:
        collection = get_mongo_collection()
        # Retrieve the last 24 records sorted by timestamp descending
        mongo_records = list(collection.find().sort("timestamp", -1).limit(24))
        
        if len(mongo_records) >= 6:
            # Load historical documents into a DataFrame
            hist_df = pd.DataFrame(mongo_records)
            
            # Compute rolling features safely from true historical data
            pm25_roll_6h = float(hist_df['pm2_5'].iloc[:6].mean())
            pm25_roll_24h = float(hist_df['pm2_5'].iloc[:len(hist_df)].mean())
            
            # Rate of change between the newest item and 4 intervals ago
            pm25_change_rate = float((hist_df['pm2_5'].iloc[0] - hist_df['pm2_5'].iloc[3]) / (hist_df['pm2_5'].iloc[3] + 1e-5))
        else:
            # Fallback to current API computation if MongoDB collection is empty/new
            st.info("💡 Populating fallback metrics from live data array stream...")
            pm25_roll_6h = float(live_df['pm2_5'].iloc[-6:].mean())
            pm25_roll_24h = float(live_df['pm2_5'].iloc[-24:].mean())
            pm25_change_rate = float((live_df['pm2_5'].iloc[-1] - live_df['pm2_5'].iloc[-4]) / (live_df['pm2_5'].iloc[-4] + 1e-5))
            
    except Exception as db_err:
        st.caption(f"MongoDB Query warning (falling back to local compute arrays): {db_err}")
        pm25_roll_6h = float(live_df['pm2_5'].iloc[-6:].mean())
        pm25_roll_24h = float(live_df['pm2_5'].iloc[-24:].mean())
        pm25_change_rate = float((live_df['pm2_5'].iloc[-1] - live_df['pm2_5'].iloc[-4]) / (live_df['pm2_5'].iloc[-4] + 1e-5))

# 3. Load Local Version-Controlled Model and Make Predictions
st.subheader("🔮 3-Day Lookahead Prediction")

@st.cache_resource
def load_local_ai_model():
    """Loads the trained machine learning model artifact stored in the workspace."""
    model_filename = "aqi_model.pkl"
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Could not find model artifact: '{model_filename}'. Run training_pipeline.py first to generate it.")
    return joblib.load(model_filename)

try:
    with st.spinner("Loading trained AI model parameters from workspace repository..."):
        model = load_local_ai_model()
    
    # Engineer our runtime input payload to match what the model learned during training
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
        'pm2_5_roll_6h': pm25_roll_6h,
        'pm2_5_roll_24h': pm25_roll_24h,
        'pm2_5_change_rate': pm25_change_rate
    }])
    
    # 🔥 FORCE EXACT SAME ALPHABETICAL COLUMN ORDERING FOR INFERENCE
    inference_payload = inference_payload.reindex(sorted(inference_payload.columns), axis=1)
    
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
    
    # 4. 📉 Advanced Analytics & Model Interpretability (SHAP)
    st.subheader("📊 Advanced Analytics: Feature Importance (SHAP)")
    st.markdown("This section calculates which weather features and indicators matter most to the AI model's forecast computations.")
    
    with st.spinner("Calculating SHAP mathematical dependency maps..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(inference_payload)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[0])
        else:
            vals = np.abs(shap_values)
            
        importance_df = pd.DataFrame({
            'Feature': inference_payload.columns,
            'SHAP Importance': vals[0]
        }).sort_values(by='SHAP Importance', ascending=True)
        
        ax.barh(importance_df['Feature'], importance_df['SHAP Importance'], color='#4682B4')
        ax.set_xlabel('Absolute Impact Score on 3-Day Forecast Prediction')
        plt.tight_layout()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Failed to compile dashboard parameters or SHAP charts: {e}")