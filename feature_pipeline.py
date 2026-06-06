import os
import tempfile
import requests
import pandas as pd
import hopsworks
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Set your Hopsworks API key here so the script can log in automatically
os.environ["HOPSWORKS_API_KEY"] = "nqwhi0tLZlZzJQpq.jQ0OUyguCdCUbb11UoH4HA8qHmJmXyna27JEPJwJcszSet2W5GNRRopC7WpQZGSz"

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
    
    # Convert raw JSON data into a clean spreadsheet/dataframe format
    df = pd.DataFrame(raw_data)
    df['time'] = pd.to_datetime(df['time'])
    return df


def engineer_features(df):
    """Step 2: Transform raw numbers into meaningful patterns (Features)"""
    print("🧹 Engineering features and creating target variables...")
    
    # Ensure data is sorted correctly by time order
    df = df.sort_values('time').reset_index(drop=True)
    
    # Time-based features: Models love learning hour/month patterns!
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Derived features: Calculate rolling averages over the last 6 and 24 hours
    df['pm2_5_roll_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    df['pm2_5_roll_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    # AQI Change Rate: How fast is PM2.5 rising or falling compared to 3 hours ago?
    df['pm2_5_change_rate'] = df['pm2_5'].pct_change(periods=3).fillna(0.0)
    
    # THE TARGET: We want to predict what PM2.5 will be 3 Days (72 Hours) into the future
    # We "shift" the data backwards by 72 rows so today's features align with the future value
    df['target_pm2_5'] = df['pm2_5'].shift(-72)
    
    # Hopsworks requires a string format timestamp or simple ID column as a Primary Key
    df['timestamp'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['time'])
    
    # Drop rows where target is missing (this happens at the very end of our dataset due to the 72h shift)
    df = df.dropna(subset=['target_pm2_5'])
    return df


def upload_to_hopsworks(df):
    """Step 3: Connect to Hopsworks and save our data table"""
    print("🔒 Logging into Hopsworks Store...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        project="Karachi_Weather_Forecast"
    )
    fs = project.get_feature_store()
    
    print("📦 Creating/Updating Feature Group in the cloud...")
    aqi_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=1,
        primary_key=['timestamp'],
        description="Hourly engineered air quality variables for Karachi",
        online_enabled=True
    )
    
    # Upload the dataframe. write_options={"wait_for_job": False} lets GitHub finish 
    # instantly while Hopsworks runs its background ingestion Spark job asynchronously.
    aqi_fg.insert(df, write_options={"wait_for_job": False})
    print("🎉 Feature Pipeline Successful! Check your Hopsworks UI dashboard.")


if __name__ == "__main__":
    # Dynamically calculate a moving 3-month (90 days) lookback timeline window
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    raw_dataframe = fetch_raw_data(start, end)
    processed_dataframe = engineer_features(raw_dataframe)
    upload_to_hopsworks(processed_dataframe)