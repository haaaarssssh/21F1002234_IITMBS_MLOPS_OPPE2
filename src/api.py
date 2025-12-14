from fastapi import FastAPI, HTTPException
from pantic import BaseModel, Field
import joblib
import pandas as pd
import gcsfs
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("--- API STARTING UP ---")

app = FastAPI(title="Heart Disease Prediction API")

# --- Load Model ---
model = None
PROJECT_ID = os.environ.get("GCP_PROJECT")
BUCKET_NAME = f"gs://{PROJECT_ID}-oppe2-bucket"
MODEL_PATH = "model/heart_disease_model.joblib"

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Attempting to load model from GCS path: {BUCKET_NAME}/{MODEL_PATH}")
        print(f"Using Project ID: {PROJECT_ID}")
        
        fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
        print("GCSFileSystem object created successfully.")

        with fs.open(f"{BUCKET_NAME}/{MODEL_PATH}", "rb") as f:
            print("File opened from GCS, starting to load model with joblib...")
            model = joblib.load(f)
            print("!!! Model loaded successfully. !!!")

    except Exception as e:
        print(f"!!! FATAL ERROR DURING MODEL LOADING: {e} !!!")
        # In a real app, you would raise this, but for debugging, we print
        # raise RuntimeError(f"Could not load model from GCS: {e}")

# The rest of your api.py file (PredictionInput, PredictionOutput, endpoints) remains the same...
# --- API Data Model ---
class PredictionInput(BaseModel):
    sno: int = Field(..., example=87)
    age: int = Field(..., example=46)
    gender: int = Field(..., example=0, description="0 for male, 1 for female")
    cp: int = Field(..., example=1)
    trestbps: float = Field(..., example=101.0)
    chol: float = Field(..., example=197.0)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=1)
    thalach: float = Field(..., example=156.0)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=0.0)
    slope: int = Field(..., example=2)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=3)

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Heart Disease Prediction API is running."}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction_result = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        label = "Heart Disease" if prediction_result == 0 else "No Heart Disease"

        # Log the prediction
        logger.info(f"Prediction successful for sno {input_data.sno}. Result: {label}, Probs: {probability}")
        
        return {"prediction": int(prediction_result), "prediction_label": label}
    except Exception as e:
        logger.error(f"Prediction failed for sno {input_data.sno}. Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
