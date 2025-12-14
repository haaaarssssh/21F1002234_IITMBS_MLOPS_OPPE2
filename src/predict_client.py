import pandas as pd
import numpy as np
import requests
import time
import os

# Get API IP from environment
API_IP = os.environ.get("API_IP")
if not API_IP:
    raise ValueError("API_IP environment variable not set.")

URL = f"http://{API_IP}/predict"

# Load original data to understand schema and ranges
df = pd.read_csv("data/data.csv")
df['gender'] = pd.factorize(df['gender'])[0] # Match encoding

# Generate 100 random samples based on original data stats
num_samples = 100
random_data = {}
for col in df.columns:
    if col not in ['target', 'sno']:
        if df[col].dtype in ['int64', 'float64']:
            min_val, max_val = df[col].min(), df[col].max()
            if df[col].dtype == 'int64':
                random_data[col] = np.random.randint(min_val, max_val + 1, num_samples)
            else:
                random_data[col] = np.random.uniform(min_val, max_val, num_samples)

random_df = pd.DataFrame(random_data)
random_df['sno'] = range(1000, 1000 + num_samples) # Add unique sno
random_df.to_csv("data/generated_data.csv", index=False)
print("Generated 100 random samples and saved to data/generated_data.csv")

# Send requests
for i, row in random_df.iterrows():
    payload = row.to_dict()
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        print(f"sno: {payload['sno']}, Status: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"sno: {payload['sno']}, Error: {e}")
    time.sleep(0.1) # Avoid overwhelming the API
