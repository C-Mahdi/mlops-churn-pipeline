"""
FastAPI Application for Churn Prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager

# Global variables to store loaded models
model = None
scaler = None
encoder_state = None
encoder_area = None
columns_order = None

# Configuration
MODEL_PATH = "models/churn_model.joblib"
SCALER_PATH = "models/scaler.joblib"
ENCODER_STATE_PATH = "models/encoder_state.joblib"
ENCODER_AREA_PATH = "models/encoder_area.joblib"
COLUMNS_ORDER_PATH = "models/columns_order.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global model, scaler, encoder_state, encoder_area, columns_order
    
    print("Loading models...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder_state = joblib.load(ENCODER_STATE_PATH)
        encoder_area = joblib.load(ENCODER_AREA_PATH)
        columns_order = joblib.load(COLUMNS_ORDER_PATH)
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please train and save models first using the pipeline.")
    
    yield
    
    print("Shutting down...")


app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using trained ML model",
    version="1.0.0",
    lifespan=lifespan
)


class CustomerData(BaseModel):
    """Input schema for customer data"""
    State: str = Field(..., example="NY")
    Account_length: int = Field(..., alias="Account length", example=128)
    Area_code: int = Field(..., alias="Area code", example=415)
    International_plan: str = Field(..., alias="International plan", example="No")
    Voice_mail_plan: str = Field(..., alias="Voice mail plan", example="Yes")
    Number_vmail_messages: int = Field(..., alias="Number vmail messages", example=25)
    Total_day_minutes: float = Field(..., alias="Total day minutes", example=265.1)
    Total_day_calls: int = Field(..., alias="Total day calls", example=110)
    Total_day_charge: float = Field(..., alias="Total day charge", example=45.07)
    Total_eve_minutes: float = Field(..., alias="Total eve minutes", example=197.4)
    Total_eve_calls: int = Field(..., alias="Total eve calls", example=99)
    Total_eve_charge: float = Field(..., alias="Total eve charge", example=16.78)
    Total_night_minutes: float = Field(..., alias="Total night minutes", example=244.7)
    Total_night_calls: int = Field(..., alias="Total night calls", example=91)
    Total_night_charge: float = Field(..., alias="Total night charge", example=11.01)
    Total_intl_minutes: float = Field(..., alias="Total intl minutes", example=10.0)
    Total_intl_calls: int = Field(..., alias="Total intl calls", example=3)
    Total_intl_charge: float = Field(..., alias="Total intl charge", example=2.7)
    Customer_service_calls: int = Field(..., alias="Customer service calls", example=1)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Output schema for prediction response"""
    churn_prediction: int
    churn_probability: float
    confidence: float
    risk_level: str


class BatchCustomerData(BaseModel):
    """Input schema for batch predictions"""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_percentage: float


def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """Preprocess input data to match training pipeline"""
    
    # Convert to dictionary and then DataFrame
    data_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([data_dict])
    
    # Encode binary features
    binary_cols = ["International plan", "Voice mail plan"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})
    
    # Encode State using saved encoder
    encoded_states = encoder_state.transform(df[["State"]])
    encoded_states_df = pd.DataFrame(
        encoded_states,
        columns=encoder_state.get_feature_names_out(["State"])
    )
    
    # Encode Area code using saved encoder
    encoded_area = encoder_area.transform(df[["Area code"]])
    encoded_area_df = pd.DataFrame(
        encoded_area,
        columns=encoder_area.get_feature_names_out(["Area code"])
    )
    
    # Drop original categorical columns
    df = df.drop(["State", "Area code"], axis=1)
    
    # Concatenate encoded features
    df = pd.concat([df, encoded_states_df, encoded_area_df], axis=1)
    
    # Feature engineering
    df = create_engineered_features(df)
    
    # Drop correlated features
    df = drop_correlated_features(df)
    
    # Ensure column order matches training
    df = df.reindex(columns=columns_order, fill_value=0)
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features"""
    df["Total calls"] = (
        df["Total day calls"]
        + df["Total eve calls"]
        + df["Total night calls"]
        + df["Total intl calls"]
    )
    
    df["Total charge"] = (
        df["Total day charge"]
        + df["Total eve charge"]
        + df["Total night charge"]
        + df["Total intl charge"]
    )
    
    df["CScalls Rate"] = df["Customer service calls"] / (df["Account length"] + 1)
    
    return df


def drop_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop correlated features"""
    correlated_cols = [
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
        "Voice mail plan",
    ]
    
    return df.drop(columns=[col for col in correlated_cols if col in df.columns])


def get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single customer prediction",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        model is not None,
        scaler is not None,
        encoder_state is not None,
        encoder_area is not None,
        columns_order is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn for a single customer
    
    Args:
        customer: Customer data following the CustomerData schema
        
    Returns:
        Prediction response with churn prediction and probability
    """
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and saved."
        )
    
    try:
        # Preprocess input
        X_processed = preprocess_input(customer)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0, 1]
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2
        
        # Determine risk level
        risk_level = get_risk_level(probability)
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            confidence=float(confidence),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(batch_data: BatchCustomerData):
    """
    Predict churn for multiple customers
    
    Args:
        batch_data: List of customers
        
    Returns:
        Batch prediction response with all predictions
    """
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and saved."
        )
    
    try:
        predictions = []
        churn_count = 0
        
        for customer in batch_data.customers:
            # Preprocess input
            X_processed = preprocess_input(customer)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            probability = model.predict_proba(X_processed)[0, 1]
            
            # Calculate confidence
            confidence = abs(probability - 0.5) * 2
            
            # Determine risk level
            risk_level = get_risk_level(probability)
            
            predictions.append(PredictionResponse(
                churn_prediction=int(prediction),
                churn_probability=float(probability),
                confidence=float(confidence),
                risk_level=risk_level
            ))
            
            if prediction == 1:
                churn_count += 1
        
        total_customers = len(batch_data.customers)
        churn_percentage = (churn_count / total_customers) * 100 if total_customers > 0 else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=total_customers,
            churn_count=churn_count,
            churn_percentage=float(churn_percentage)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)