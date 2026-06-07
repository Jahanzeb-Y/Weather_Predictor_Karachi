# Karachi 3-Day AQI Prediction Engine
An automated, real-time Machine Learning pipeline and interactive analytical dashboard designed to forecast Air Quality Index (US AQI) trajectories up to 72 hours out for Karachi.

# Key Highlights
✅ Live automated feature ingestion from distributed MongoDB storage
✅ multi-model architecture with interactive framework toggling
✅ Interactive responsive Streamlit dashboard with real-time 72-hour Plotly forecasts
✅ Interpretability matrix via Explainable AI (SHAP Framework)

# Tech Stack
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainable AI:** SHAP Framework, Matplotlib
* **Visualization:** Streamlit UI, Plotly Graph Objects
* **Infrastructure & Database:** MongoDB Atlas (Cloud Database)

# System Architecture
The application runs a continuous end-to-end data processing and inference lifecycle:
1. **Data Ingestion:** Automatically polls hourly and daily environmental telemetry from a cloud-hosted MongoDB collection (`Karachi_Weather_Forecast`).
2. **Feature Engineering:** Computes rolling averages (`pm2_5_roll_6h`, `pm2_5_roll_24h`) and momentum metrics (`pm2_5_change_rate`) on the fly.
3. **Multi-Model Inference:** Allows dynamic toggling between models to predict.
4. **Explainable AI (XAI):** Implements localized mathematical feature scoring via `SHAP` matrices to illustrate feature impacts on the active prediction record.

---

## Pipeline Evaluation Matrix
Validation performance scores calculated across models from historical training partitions:
| Forecasting Framework | RMSE (Error) | MAE (Error) | R² Accuracy Score |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost Engine** | **5.9681** | **4.2454** | **0.5963** | **Current Production Leader** 
| Random Forest Ensembles | 6.4145 | 4.4829 | 0.5337 
| Ridge Linear Regression | 9.3212 | 6.9137 | 0.0153

---

## Repository Directory Tree
aqi_forecaster/
├── app.py                  # Streamlit Dashboard 
├── train_pipeline.py       # Training Routine
├── requirements.txt        # Environment Package
├── model_metrics.txt       # Metric Execution Logs
├── .gitignore              # Security Exclusions
├── aqi_model_xgb.pkl       # XGBoost Model
├── aqi_model_rf.pkl        # Random Forest Model
└── aqi_model_ridge.pkl     # Ridge Linear Regression

# Setup & Installation
1. Clone the Repository
git clone [https://github.com/Jahanzeb-Y/Weather_Predictor_Karachi.git](https://github.com/Jahanzeb-Y/Weather_Predictor_Karachi.git)
cd Weather_Predictor_Karachi

2. Create Virtual Environment
python -m venv venv
# Windows Activation
.\venv\Scripts\activate
# Mac/Linux Activation
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
MONGO_URI

5. Usage
streamlit run app.py


