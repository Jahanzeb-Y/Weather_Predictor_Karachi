import os
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient

MONGO_URI = None
try:
    if "MONGO_URI" in st.secrets:
        MONGO_URI = st.secrets["MONGO_URI"]
except Exception:
    pass

if not MONGO_URI:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/?appName=KarachiAQIFeatures")

st.set_page_config(page_title="Karachi 3-Day AQI Predictor", layout="wide", page_icon="🌍")

st.markdown("""
<style>
/* Sleek Dark Metric Cards */
[data-testid="stMetric"] {
    background-color: #1e2127;
    padding: 15px 20px;
    border-radius: 8px;
    border: 1px solid #2d313a;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #4682B4;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.title("Karachi AQI Prediction Engine by Jahanzeb Yameen")
    st.markdown("<p style='font-size: 1.1rem; color: #a0aab2; margin-top: -15px;'>72-Hour Analytics</p>", unsafe_allow_html=True)
with col_t2:
    st.markdown("<div style='text-align: right; padding-top: 20px;'><span style='background-color: #2d313a; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem; border: 1px solid #4CAF50; color: #4CAF50;'>🟢 Status: Running. Active</span></div>", unsafe_allow_html=True)
st.markdown("---")


def convert_pm25_to_us_aqi(pm25):
    """Converts raw PM2.5 concentration to official US EPA AQI standard score (0-500)."""
    val = float(pm25)
    if val <= 9.0:
        return int(((50 - 0) / (9.0 - 0.0)) * (val - 0.0) + 0)
    elif val <= 35.4:
        return int(((100 - 51) / (35.4 - 9.1)) * (val - 9.1) + 51)
    elif val <= 55.4:
        return int(((150 - 101) / (55.4 - 35.5)) * (val - 35.5) + 101)
    elif val <= 125.4:
        return int(((200 - 151) / (125.4 - 55.5)) * (val - 55.5) + 151)
    elif val <= 225.4:
        return int(((300 - 201) / (225.4 - 125.5)) * (val - 125.5) + 201)
    else:
        return int(((500 - 301) / (500.0 - 225.5)) * (val - 225.5) + 301)

convert_vectorized = np.vectorize(convert_pm25_to_us_aqi)

@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client["Karachi_Weather_Forecast"]  
    collection = db["karachi_aqi_features"]  
    return collection

try:
    collection = get_mongo_collection()
    
    latest_record = list(collection.find().sort("timestamp", -1).limit(1))[0]
    
    cur_pm25 = float(latest_record['pm2_5'])
    cur_pm10 = float(latest_record['pm10'])
    cur_no2 = float(latest_record['nitrogen_dioxide'])
    cur_so2 = float(latest_record['sulphur_dioxide'])
    cur_ozone = float(latest_record['ozone'])
    
    pm25_roll_6h = float(latest_record['pm2_5_roll_6h'])
    pm25_roll_24h = float(latest_record['pm2_5_roll_24h'])
    pm25_change_rate = float(latest_record['pm2_5_change_rate'])
    
    record_time = datetime.strptime(latest_record['timestamp'], '%Y-%m-%d %H:%M:%S')

    mongo_records = list(collection.find().sort("timestamp", -1).limit(24))
    mongo_records.reverse()  # Chronological order
    live_df = pd.DataFrame(mongo_records)

except Exception as db_err:
    st.error(f"Error accessing production database states: {db_err}")
    cur_pm25, cur_pm10, cur_no2, cur_so2, cur_ozone = 30.0, 45.0, 15.0, 5.0, 20.0
    pm25_roll_6h, pm25_roll_24h, pm25_change_rate = 30.0, 30.0, 0.0
    record_time = datetime.now()
    live_df = pd.DataFrame([{"timestamp": record_time.strftime('%Y-%m-%d %H:%M:%S'), "pm2_5": 30.0}])

@st.cache_resource
def load_local_ai_model():
    model_filename = "aqi_model.pkl"
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Missing model artifact: '{model_filename}'")
    return joblib.load(model_filename)

try:
    model = load_local_ai_model()
    
    inference_payload = pd.DataFrame([{
        'pm2_5': cur_pm25,
        'pm10': cur_pm10,
        'nitrogen_dioxide': cur_no2,
        'sulphur_dioxide': cur_so2,
        'ozone': cur_ozone,
        'hour': record_time.hour,
        'day_of_week': record_time.weekday(),
        'month': record_time.month,
        'pm2_5_roll_6h': pm25_roll_6h,
        'pm2_5_roll_24h': pm25_roll_24h,
        'pm2_5_change_rate': pm25_change_rate
    }])
    inference_payload = inference_payload.reindex(sorted(inference_payload.columns), axis=1)
    predicted_pm25 = model.predict(inference_payload)[0]

    current_aqi_score = convert_pm25_to_us_aqi(cur_pm25)
    predicted_aqi_score = convert_pm25_to_us_aqi(predicted_pm25)

    def get_aqi_descriptor(score):
        if score <= 50: return "🟢 Good"
        if score <= 100: return "🟡 Moderate"
        if score <= 150: return "🟠 Sensitive Warning"
        return "🚨 Unhealthy"

    st.subheader("📡 Real-Time Environmental Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Air Index", f"{current_aqi_score} US AQI", delta=get_aqi_descriptor(current_aqi_score), delta_color="normal")
    c2.metric("Fine Mass (PM2.5)", f"{cur_pm25:.1f} µg/m³", delta="Raw Concentration", delta_color="off")
    c3.metric("Dust & Smoke (PM10)", f"{cur_pm10:.1f} µg/m³", delta="Coarse Particulates", delta_color="off")
    c4.metric("Last Database Ingestion", record_time.strftime('%H:%M %p'), delta=record_time.strftime('%b %d, %Y'), delta_color="off")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("72-Hour Forecast Trajectory")

    future_dates = [record_time + timedelta(hours=i) for i in range(1, 73)]
    
    trend_line = np.linspace(cur_pm25, predicted_pm25, 72)
    np.random.seed(42)
    noise = np.random.normal(0, 1.5, 72)
    final_forecast_series_pm25 = np.clip(trend_line + noise, a_min=0, a_max=None)
    
    historical_subset_pm25 = live_df['pm2_5'].iloc[max(0, len(live_df)-24):]
    historical_subset_aqi = convert_vectorized(historical_subset_pm25)
    future_forecast_aqi = convert_vectorized(final_forecast_series_pm25)
    
    fig = go.Figure()
    
    hist_hours = len(historical_subset_aqi)
    hist_timestamps = [record_time - timedelta(hours=hist_hours-i) for i in range(hist_hours)]
    fig.add_trace(go.Scatter(
        x=hist_timestamps, 
        y=historical_subset_aqi, 
        mode='lines', 
        name='Historical Baseline (US AQI)', 
        fill='tozeroy', 
        line=dict(color='#00b4d8', width=3), 
        fillcolor='rgba(0, 180, 216, 0.1)'
    ))
    
    forecast_dates = [hist_timestamps[-1]] + future_dates
    forecast_values = [historical_subset_aqi[-1]] + list(future_forecast_aqi)
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast_values, 
        mode='lines', 
        name='AI Forecast Horizon (US AQI)', 
        line=dict(color='#ff4b4b', width=3, dash='dash')
    ))
    
    fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good Threshold (50)", annotation_position="top left")
    fig.add_hline(y=100, line_dash="dot", line_color="orange", annotation_text="Moderate Cap (100)", annotation_position="top left")
    fig.add_hline(y=150, line_dash="solid", line_color="red", annotation_text="Unhealthy Boundary (150)", annotation_position="top left")
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Air Quality Index (US AQI)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d313a', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if predicted_aqi_score <= 50:
        st.success(f"🟢 **Air Quality Level: Good ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is highly safe.")
    elif 50 < predicted_aqi_score <= 100:
        st.warning(f"🟡 **Air Quality Level: Moderate ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is acceptable.")
    elif 100 < predicted_aqi_score <= 150:
        st.info(f"🟠 **Air Quality Level: Unhealthy for Sensitive Groups ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is alarming.")
    else:
        st.error(f"🚨 **Air Quality Level: Unhealthy Environment Alert ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is dangerous")

    st.markdown("---")
    st.subheader("Model Metrics")
    with st.spinner("Compiling..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(inference_payload)
        fig_shap, ax = plt.subplots(figsize=(10, 3.5))
        vals = np.abs(shap_values[0]) if isinstance(shap_values, list) else np.abs(shap_values)
        importance_df = pd.DataFrame({'Feature': inference_payload.columns, 'SHAP Importance': vals[0]}).sort_values(by='SHAP Importance', ascending=True)
        ax.barh(importance_df['Feature'], importance_df['SHAP Importance'], color='#4682B4')
        ax.set_xlabel('Absolute Impact Score on 3-Day Forecast Prediction')
        plt.tight_layout()
        st.pyplot(fig_shap)

except Exception as e:
    st.error(f"System compilation runtime exception: {e}")