import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import joblib
import gcsfs
import os

# --- Configuration ---
# In a real app, use src/config.py, but for simplicity here:
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = f"gs://{PROJECT_ID}-oppe2-bucket"
MODEL_PATH = "model/heart_disease_model.joblib"
DATA_PATH = "data/data.csv"

def preprocess_data(df):
    """Cleans and preprocesses the data."""
    # Handle categorical features
    df['gender'] = pd.factorize(df['gender'])[0]
    df['target'] = pd.factorize(df['target'])[0] # 'yes' -> 0, 'no' -> 1

    # Drop rows with any missing values
    df.dropna(inplace=True)
    return df

def train_model():
    """Loads data, preprocesses, trains, and saves the model."""
    print("Starting model training...")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data with {df.shape[0]} rows.")

    # Preprocess data
    cleaned_df = preprocess_data(df)
    print(f"Data cleaned. {cleaned_df.shape[0]} rows remaining.")

    X = cleaned_df.drop("target", axis=1)
    y = cleaned_df["target"]

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training with hyperparameter tuning
    log_reg_grid = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["liblinear"]
    }
    rs_log_reg = RandomizedSearchCV(
        LogisticRegression(max_iter=1000),
        param_distributions=log_reg_grid,
        cv=5,
        n_iter=20,
        verbose=1,
        random_state=42
    )
    print("Fitting RandomizedSearchCV...")
    rs_log_reg.fit(X_train, y_train)

    best_model = rs_log_reg.best_estimator_
    print(f"Best parameters found: {rs_log_reg.best_params_}")

    # Save model locally first
    local_model_path = "heart_disease_model.joblib"
    joblib.dump(best_model, local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # Upload model to GCS
    gcs_path = f"{BUCKET_NAME}/{MODEL_PATH}"
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    fs.put(local_model_path, gcs_path)

    print(f"Model successfully uploaded to {gcs_path}")

if __name__ == "__main__":
    train_model()