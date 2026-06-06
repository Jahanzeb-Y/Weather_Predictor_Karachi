import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient, UpdateOne

# --- CONFIGURATION ---
# We will pull this securely from GitHub Secrets in production
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://JahanzebYameen:10603770569@karachiaqifeatures.cmueb2n.mongodb.net/?appName=KarachiAQIFeatures")

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011


def fetch_raw_data(start_date, end_date):
    """Step 1: Fetch raw data from Open-Meteo API"""
    print(f"📡 Fetching data from Open-Meteo from {start_date} to {end_date}...")
    
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone&"
        f"start_date={start_date}&end_date={end_date}"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")
        
    raw_data = response.json()["hourly"]
    
    df = pd.DataFrame(raw_data)
    df['time'] = pd.to_datetime(df['time'])
    return df


def engineer_features(df):
    """Step 2: Transform raw numbers into meaningful patterns (Features)"""
    print("🧹 Engineering features and creating target variables...")
    
    # Ensure data is sorted correctly by time order
    df = df.sort_values('time').reset_index(drop=True)
    
    # Time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Derived features: Calculate rolling averages
    df['pm2_5_roll_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    df['pm2_5_roll_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    # AQI Change Rate
    df['pm2_5_change_rate'] = df['pm2_5'].pct_change(periods=3).fillna(0.0)
    
    # THE TARGET: Predict 3 Days (72 Hours) into the future
    df['target_pm2_5'] = df['pm2_5'].shift(-72)
    
    # Standard string timestamp formatting for document indexing
    df['timestamp'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['time'])
    
    # Drop rows where target is missing
    df = df.dropna(subset=['target_pm2_5'])
    return df


def upload_to_mongodb(df):
    """Step 3: Connect to MongoDB and bulk upsert records safely"""
    print("🔒 Connecting to MongoDB Atlas cluster...")
    if not MONGO_URI or "PASTE_YOUR_LOCAL" in MONGO_URI:
        raise ValueError("Error: MongoDB connection URI is missing or misconfigured.")
        
    client = MongoClient(MONGO_URI)
    db = client["Karachi_Weather_Forecast"]
    collection = db["karachi_aqi_features"]
    
    # Convert dataframe to a list of dictionary documents
    records = df.to_dict(orient="records")
    print(f"📦 Preparing to process {len(records)} entries...")
    
    # Create bulk upsert operations (updates if timestamp exists, inserts if new)
    operations = [
        UpdateOne({"timestamp": record["timestamp"]}, {"$set": record}, upsert=True)
        for record in records
    ]
    
    if operations:
        result = collection.bulk_write(operations)
        print(f"🎉 MongoDB Pipeline Successful!")
        print(f"   - Upserted/Updated: {result.upserted_count + result.modified_count} records.")
    else:
        print("⏸️ No new operations to process.")
        
    client.close()


if __name__ == "__main__":
    # Dynamically calculate a moving 3-month (90 days) lookback timeline window
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    raw_dataframe = fetch_raw_data(start, end)
    processed_dataframe = engineer_features(raw_dataframe)
    upload_to_mongodb(processed_dataframe)