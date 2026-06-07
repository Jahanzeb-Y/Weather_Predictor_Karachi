import os
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  

MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/Karachi_Weather_Forecast?retryWrites=true&w=majority&tls=true"
)

def run_model_experimentation():
    print("Connecting to MongoDB...")
        
    client = MongoClient(MONGO_URI)
    db = client["Karachi_Weather_Forecast"]
    collection = db["karachi_aqi_features"]
    
    print("Fetching records from MongoDB...")
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        raise Exception("Database is empty.")
        
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
        
    df = df.dropna(subset=['target_pm2_5'])
    
    print(f"Loaded {len(df)} records")
    
    X = df.drop(columns=['target_pm2_5', 'timestamp', 'date'], errors='ignore')
    X = X.reindex(sorted(X.columns), axis=1)
    
    y = df['target_pm2_5']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset split verified: {len(X_train)} training rows, {len(X_test)} verification rows.")
    print("-" * 50)
    

    print("[1/3]: Training Random Forest.")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_r2 = float(r2_score(y_test, rf_preds))
    print(f"Random Forest -> RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f} | R²: {rf_r2:.2f}")
    joblib.dump(rf_model, "aqi_model_rf.pkl")
    
 
    print("\n[2/3]: Training XGBoost.")
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
    xgb_mae = float(mean_absolute_error(y_test, xgb_preds))
    xgb_r2 = float(r2_score(y_test, xgb_preds))
    print(f"XGBoost Engine -> RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f} | R²: {xgb_r2:.2f}")
    joblib.dump(xgb_model, "aqi_model_xgb.pkl")
    

    print("\n[3/3]: Training Ridge Regression Model.")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)
    
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
    ridge_mae = float(mean_absolute_error(y_test, ridge_preds))
    ridge_r2 = float(r2_score(y_test, ridge_preds))
    print(f"Ridge Regression -> RMSE: {ridge_rmse:.2f} | MAE: {ridge_mae:.2f} | R²: {ridge_r2:.2f}")
    joblib.dump(ridge_model, "aqi_model_ridge.pkl")
    
    print("-" * 50)
    print("\nEvaluating Metrics.")
    
    maes = {"Random Forest": rf_mae, "XGBoost": xgb_mae, "Ridge Regression": ridge_mae}
    chosen_algo = min(maes, key=maes.get)
    print(f"Statistical Champion: {chosen_algo}!")
    
    if chosen_algo == "Random Forest":
        joblib.dump(rf_model, "aqi_model.pkl")
    elif chosen_algo == "XGBoost":
        joblib.dump(xgb_model, "aqi_model.pkl")
    else:
        joblib.dump(ridge_model, "aqi_model.pkl")
        
    with open("model_metrics.txt", "w") as f:
        f.write(f"Pipeline Evaluation Summary\n")
        f.write(f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random Forest  -> RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | R2: {rf_r2:.4f}\n")
        f.write(f"XGBoost Engine -> RMSE: {xgb_rmse:.4f} | MAE: {xgb_mae:.4f} | R2: {xgb_r2:.4f}\n")
        f.write(f"Ridge Linear   -> RMSE: {ridge_rmse:.4f} | MAE: {ridge_mae:.4f} | R2: {ridge_r2:.4f}\n")
        f.write(f"Current Deployment Leader: {chosen_algo}\n")
        
    client.close()
    print("All model artifacts generated.")

if __name__ == "__main__":
    run_model_experimentation()