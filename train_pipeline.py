import os
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/Karachi_Weather_Forecast?retryWrites=true&w=majority&tls=true"
)
def run_model_experimentation():
    print("Connecting to MongoDB")
        
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
    
    print(f"Loaded {len(df)} records with valid target values. Proceeding to matrix splits...")
    
    X = df.drop(columns=['target_pm2_5', 'timestamp', 'date'], errors='ignore')
    X = X.reindex(sorted(X.columns), axis=1)
    
    y = df['target_pm2_5']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[1/2]: Training Random Forest Ensemble...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_r2 = float(r2_score(y_test, rf_preds))
    print(f"Random Forest -> RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f} | R²: {rf_r2:.2f}")
    
    print("\n[2/2]: Training Ridge Regression Model...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)
    
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
    ridge_mae = float(mean_absolute_error(y_test, ridge_preds))
    ridge_r2 = float(r2_score(y_test, ridge_preds))
    print(f"Ridge Regression -> RMSE: {ridge_rmse:.2f} | MAE: {ridge_mae:.2f} | R²: {ridge_r2:.2f}")
    
    print("\nEvaluating Champion Performance...")
    if rf_rmse < ridge_rmse:
        print(" Winner: Random Forest Regressor!")
        champion_model = rf_model
        chosen_algo = "Random Forest"
    else:
        print("Winner: Ridge Regression Model!")
        champion_model = ridge_model
        chosen_algo = "Ridge Regression"
        
    model_filename = "aqi_model.pkl"
    joblib.dump(champion_model, model_filename)
    print(f"Model artifact saved successfully as: {model_filename}")
    
    with open("model_metrics.txt", "w") as f:
        f.write(f"Algorithm: {chosen_algo}\n")
        f.write(f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RMSE: {min(rf_rmse, ridge_rmse):.4f}\n")
        f.write(f"R2 Score: {rf_r2 if rf_rmse < ridge_rmse else ridge_r2:.4f}\n")
        
    client.close()
    print("Champion model is updated locally.")

if __name__ == "__main__":
    run_model_experimentation()