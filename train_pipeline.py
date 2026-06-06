import os
import hopsworks
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
os.environ["HOPSWORKS_API_KEY"] = "nqwhi0tLZlZzJQpq.jQ0OUyguCdCUbb11UoH4HA8qHmJmXyna27JEPJwJcszSet2W5GNRRopC7WpQZGSz"

def run_model_experimentation():
    print("🔒 Logging into Hopsworks Store...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project="Karachi_Weather_Forecast"
    )
    fs = project.get_feature_store()
    
    # 1. Pull features directly from our cloud storage table
    print("📥 Fetching engineered air quality records from the cloud...")
    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df = aqi_fg.read()
    
    # Clean features and isolate target matrix splits
    X = df.drop(columns=['target_pm2_5', 'timestamp'])
    y = df['target_pm2_5']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Experiment 1: Random Forest Regressor
    print("\n🏋️‍♂️ Experiment [1/2]: Training Random Forest Ensemble...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_r2 = float(r2_score(y_test, rf_preds))
    print(f"📊 Random Forest -> RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f} | R²: {rf_r2:.2f}")
    
    # 3. Experiment 2: Ridge Regression (Statistical Linear Model)
    print("\n🏋️‍♂️ Experiment [2/2]: Training Ridge Regression Model...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)
    
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
    ridge_mae = float(mean_absolute_error(y_test, ridge_preds))
    ridge_r2 = float(r2_score(y_test, ridge_preds))
    print(f"📊 Ridge Regression -> RMSE: {ridge_rmse:.2f} | MAE: {ridge_mae:.2f} | R²: {ridge_r2:.2f}")
    
# 4. Champion Selection Strategy (Lower RMSE Wins!)
    print("\n🏆 Evaluating Champion Performance...")
    if rf_rmse < ridge_rmse:
        print("🥇 Winner: Random Forest Regressor!")
        champion_model = rf_model
        best_metrics = {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2} # <-- Strictly numbers!
        chosen_algo = "Random Forest"
    else:
        print("🥇 Winner: Ridge Regression Model!")
        champion_model = ridge_model
        best_metrics = {"rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2} # <-- Strictly numbers!
        chosen_algo = "Ridge Regression"
        
    # 5. Save the winning binary locally
    model_filename = "aqi_model.pkl"
    joblib.dump(champion_model, model_filename)
    
    # 6. Push the optimized artifact down to Hopsworks Model Registry
    print("🚀 Uploading Champion Model to Hopsworks Cloud Registry...")
    mr = project.get_model_registry()
    
    hopsworks_model = mr.python.create_model(
        name="aqi_prediction_model",
        metrics=best_metrics, # Pass the clean numeric values dictionary
        description=f"Top-performing algorithm ({chosen_algo}) for Karachi 3-Day PM2.5 forecasting." # Text goes safely here!
    )
    hopsworks_model.save(model_filename)
    print("🎉 Experimentation Complete! Champion model is securely stored.")

if __name__ == "__main__":
    run_model_experimentation()