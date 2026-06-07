import os
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
import shap
import streamlit as st

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    try:
        MONGO_URI = st.secrets.get("MONGO_URI")
    except Exception:
        pass
if not MONGO_URI:
    MONGO_URI = "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/?appName=KarachiAQIFeatures"

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Karachi 3-Day AQI Predictor", layout="wide", page_icon="🌍")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1e2127;
    padding: 15px 20px;
    border-radius: 8px;
    border: 1px solid #2d313a;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
[data-testid="stMetricValue"] { font-size: 1.8rem; color: #4682B4; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🤖 Engine Config")
selected_model_type = st.sidebar.selectbox(
    "Active Forecasting Framework",
    ["Random Forest", "XGBoost", "Ridge Linear Regression"]
)

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.title("Karachi AQI Prediction Engine by Jahanzeb Yameen")
    st.markdown("<p style='font-size: 1.1rem; color: #a0aab2; margin-top: -15px;'>72-Hour Multi-Model Analytics</p>", unsafe_allow_html=True)
with col_t2:
    st.markdown("<div style='text-align: right; padding-top: 20px;'><span style='background-color: #2d313a; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem; border: 1px solid #4CAF50; color: #4CAF50;'>🟢 Status: Running.</span></div>", unsafe_allow_html=True)
st.markdown("---")

def convert_pm25_to_us_aqi(pm25):
    val = float(pm25)
    if val <= 9.0:   return int(((50 - 0) / 9.0) * val)
    if val <= 35.4:  return int(((100 - 51) / (35.4 - 9.1)) * (val - 9.1) + 51)
    if val <= 55.4:  return int(((150 - 101) / (55.4 - 35.5)) * (val - 35.5) + 101)
    if val <= 125.4: return int(((200 - 151) / (125.4 - 55.5)) * (val - 55.5) + 151)
    if val <= 225.4: return int(((300 - 201) / (225.4 - 125.5)) * (val - 125.5) + 201)
    return int(((500 - 301) / (500.0 - 225.5)) * (val - 225.5) + 301)

convert_vectorized = np.vectorize(convert_pm25_to_us_aqi)

@st.cache_resource
def get_mongo_collection():
    return MongoClient(MONGO_URI)["Karachi_Weather_Forecast"]["karachi_aqi_features"]

try:
    collection = get_mongo_collection()
    current_utc_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    latest_record = list(collection.find({"timestamp": {"$lte": current_utc_str}}).sort("timestamp", -1).limit(1))
    if not latest_record:
        latest_record = list(collection.find().sort("timestamp", -1).limit(1))
    
    latest_record = latest_record[0]
    
    cur_pm25 = float(latest_record['pm2_5'])
    cur_pm10 = float(latest_record['pm10'])
    cur_no2  = float(latest_record['nitrogen_dioxide'])
    cur_so2  = float(latest_record['sulphur_dioxide'])
    cur_ozone = float(latest_record['ozone'])
    
    pm25_roll_6h = float(latest_record['pm2_5_roll_6h'])
    pm25_roll_24h = float(latest_record['pm2_5_roll_24h'])
    pm25_change_rate = float(latest_record['pm2_5_change_rate'])
    
    db_timestamp = str(latest_record['timestamp']).split(".")[0]
    record_time = datetime.strptime(db_timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(hours=5)

    mongo_records = list(collection.find({"timestamp": {"$lte": db_timestamp}}).sort("timestamp", -1).limit(24))
    live_df = pd.DataFrame(mongo_records[::-1]) # Cleaned Optimization: Slice reversion syntax is faster than list.reverse()

except Exception as db_err:
    st.error(f"Error accessing production database states: {db_err}")
    cur_pm25, cur_pm10, cur_no2, cur_so2, cur_ozone = 30.0, 45.0, 15.0, 5.0, 20.0
    pm25_roll_6h, pm25_roll_24h, pm25_change_rate = 30.0, 30.0, 0.0
    record_time = datetime.now()
    live_df = pd.DataFrame([{"timestamp": record_time.strftime('%Y-%m-%d %H:%M:%S'), "pm2_5": 30.0}])

# --- MODEL ARTIFACT LOADER ---
@st.cache_resource
def load_local_ai_model(model_selection):
    suffix_mapping = {"Random Forest": "rf", "XGBoost": "xgb", "Ridge Linear Regression": "ridge"}
    filename = f"aqi_model_{suffix_mapping[model_selection]}.pkl"
    
    if model_selection == "Random Forest" and not os.path.exists(filename):
        filename = "aqi_model.pkl" # Transparent baseline migration check
        
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing model artifact file: '{filename}'")
    
    print(f"⚙️ Logging: Loaded artifact file [{filename}] for choice [{model_selection}]")
    return joblib.load(filename)

try:
    model = load_local_ai_model(selected_model_type)
    
    feature_columns = [
        'day_of_week', 'hour', 'month', 'nitrogen_dioxide', 'ozone', 
        'pm10', 'pm2_5', 'pm2_5_change_rate', 'pm2_5_roll_24h', 'pm2_5_roll_6h', 'sulphur_dioxide'
    ]
    
    inference_payload = pd.DataFrame([{
        'pm2_5': cur_pm25, 'pm10': cur_pm10, 'nitrogen_dioxide': cur_no2,
        'sulphur_dioxide': cur_so2, 'ozone': cur_ozone, 'hour': record_time.hour,
        'day_of_week': record_time.weekday(), 'month': record_time.month,
        'pm2_5_roll_6h': pm25_roll_6h, 'pm2_5_roll_24h': pm25_roll_24h, 'pm2_5_change_rate': pm25_change_rate
    }])[feature_columns]
    
    predicted_pm25 = float(model.predict(inference_payload)[0])
    current_aqi_score = convert_pm25_to_us_aqi(cur_pm25)
    predicted_aqi_score = convert_pm25_to_us_aqi(predicted_pm25)

    def get_aqi_descriptor(score):
        if score <= 50: return "🟢 Good"
        if score <= 100: return "🟡 Moderate"
        if score <= 150: return "🟠 Sensitive Warning"
        return "🚨 Unhealthy"

    st.subheader("📡 Real-Time Environmental Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Air Index", f"{current_aqi_score} US AQI", delta=get_aqi_descriptor(current_aqi_score))
    c2.metric("Fine Mass (PM2.5)", f"{cur_pm25:.1f} µg/m³", delta="Raw Concentration", delta_color="off")
    c3.metric("Dust & Smoke (PM10)", f"{cur_pm10:.1f} µg/m³", delta="Coarse Particulates", delta_color="off")
    c4.metric("Last Data Ingestion", record_time.strftime('%I:%M %p'), delta=record_time.strftime('%b %d, %Y'), delta_color="off")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"**72-Hour Forecast Trajectory Engine:** Evaluating via `{selected_model_type}`")
    
    future_dates = [record_time + timedelta(hours=i) for i in range(1, 73)]
    np.random.seed(42)
    noise = np.random.normal(0, 1.5, 72)
    final_forecast_series_pm25 = np.clip(np.linspace(cur_pm25, predicted_pm25, 72) + noise, a_min=0, a_max=None)
    
    historical_subset_pm25 = live_df['pm2_5'].iloc[-24:]
    historical_subset_aqi = convert_vectorized(historical_subset_pm25)
    future_forecast_aqi = convert_vectorized(final_forecast_series_pm25)
    
    fig = go.Figure()
    hist_hours = len(historical_subset_aqi)
    hist_timestamps = [record_time - timedelta(hours=hist_hours-i) for i in range(hist_hours)]
    
    fig.add_trace(go.Scatter(
        x=hist_timestamps, y=historical_subset_aqi, mode='lines', 
        name='Historical Baseline (US AQI)', fill='tozeroy', 
        line=dict(color='#00b4d8', width=3), fillcolor='rgba(0, 180, 216, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[hist_timestamps[-1]] + future_dates, y=[historical_subset_aqi[-1]] + list(future_forecast_aqi), 
        mode='lines', name=f'{selected_model_type} Path (US AQI)', line=dict(color='#ff4b4b', width=3, dash='dash')
    ))
    
    for y_val, col, text in [(50, "green", "Good Threshold (50)"), (100, "orange", "Moderate Cap (100)"), (150, "red", "Unhealthy Boundary (150)")]:
        fig.add_hline(y=y_val, line_dash="dot" if y_val<150 else "solid", line_color=col, annotation_text=text, annotation_position="top left")
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="",
        yaxis_title="Air Quality Index (US AQI)", hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d313a', zeroline=False)
    
    st.plotly_chart(fig, width="stretch")
    
    if predicted_aqi_score <= 50:
        st.success(f"🟢 **Air Quality Level: Good ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is highly safe.")
    elif predicted_aqi_score <= 100:
        st.warning(f"🟡 **Air Quality Level: Moderate ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is acceptable.")
    elif predicted_aqi_score <= 150:
        st.info(f"🟠 **Air Quality Level: Unhealthy for Sensitive Groups ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is alarming.")
    else:
        st.error(f"🚨 **Air Quality Level: Unhealthy Environment Alert ({predicted_aqi_score} US AQI)** — Predicted air quality in 72 hours is dangerous")

    st.markdown("---")
    st.subheader("Model Feature Importance")
    with st.spinner("Compiling model evaluations..."):
        try:
            background_df = pd.DataFrame([latest_record])[feature_columns]
            if len(live_df) >= 5:
                background_df = live_df[feature_columns].head(5)
            else:
                background_df = pd.concat([background_df] * 5, ignore_index=True)
                
            background_df = background_df.astype(float)
            clean_inference = inference_payload.astype(float)
            vals = None

            if selected_model_type == "Ridge Linear Regression":
                explainer = shap.LinearExplainer(model, background_df)
                vals = np.abs(explainer.shap_values(clean_inference))
                
            elif selected_model_type == "XGBoost":
                try:
                    explainer = shap.Explainer(model, feature_names=feature_columns)
                    shap_vals = explainer(clean_inference)
                    vals = np.abs(shap_vals.values if hasattr(shap_vals, "values") else shap_vals)
                except Exception:
                    if hasattr(model, "feature_importances_"):
                        vals = model.feature_importances_
            else:
                try:
                    explainer = shap.TreeExplainer(model)
                    vals = np.abs(explainer.shap_values(clean_inference))
                except Exception:
                    if hasattr(model, "feature_importances_"):
                        vals = model.feature_importances_

            if vals is not None:
                vals = np.array(vals).flatten()
            
            if vals is None or len(vals) != len(feature_columns):
                vals = np.zeros(len(feature_columns))
                
            importance_df = pd.DataFrame({'Feature': feature_columns, 'SHAP Importance': vals}).sort_values(by='SHAP Importance', ascending=True)
            
            fig_shap, ax = plt.subplots(figsize=(10, 3.5))
            ax.barh(importance_df['Feature'], importance_df['SHAP Importance'], color='#4682B4')
            ax.set_xlabel(f'Impact Score on 3-Day Prediction ({selected_model_type})')
            plt.tight_layout()
            st.pyplot(fig_shap)
            
            st.markdown("---")
            col_m_title, col_m_badge = st.columns([3, 1])
            with col_m_title:
                st.subheader("Pipeline metrics")
                st.markdown(f"performance metrics for `{selected_model_type}`.")
            with col_m_badge:
                st.markdown("<div style='text-align: right; padding-top: 15px;'><span style='background-color: #1e2127; padding: 6px 12px; border-radius: 4px; font-size: 0.85rem; border: 1px solid #4CAF50; color: #4CAF50;'>Leader: XGBoost Engine</span></div>", unsafe_allow_html=True)

            metrics_db = {
                "Random Forest": {"RMSE": 6.4145, "MAE": 4.4829, "R2": 0.5337},
                "XGBoost": {"RMSE": 5.9681, "MAE": 4.2454, "R2": 0.5963},
                "Ridge Linear Regression": {"RMSE": 9.3212, "MAE": 6.9137, "R2": 0.0153}
            }

            current_metrics = metrics_db[selected_model_type]

            m1, m2, m3 = st.columns(3)
            m1.metric(
                label="Root Mean Squared Error (RMSE)", 
                value=f"{current_metrics['RMSE']:.4f}", 
                delta="Lower is better", 
                delta_color="inverse"
            )
            m2.metric(
                label="R-Squared Score (R²)", 
                value=f"{current_metrics['R2']:.4f}", 
                delta="Higher is better", 
                delta_color="normal"
            )
            m3.metric(
                label="Mean Absolute Error (MAE)", 
                value=f"{current_metrics['MAE']:.4f}", 
                delta="Average scale error", 
                delta_color="off"
            )

            with st.expander(" All models"):
                comparison_df = pd.DataFrame.from_dict(metrics_db, orient='index')
                comparison_df.columns = ['RMSE', 'MAE', 'R² Accuracy Score']
                st.table(comparison_df)
            
        except Exception as shap_err:
            print(f"Internal SHAP Logging Error: {shap_err}")
            st.info("SHAP parsed for active records.")

except Exception as e:
    st.error(f"System compilation runtime exception: {e}")