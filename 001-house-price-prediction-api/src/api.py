# src/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import os
from datetime import datetime
from typing import Optional
import json

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using Random Forest model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
model_metadata = None

# Pydantic model for input validation
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    sqft_living: int = Field(..., gt=0, description="Square footage of living space")
    sqft_lot: int = Field(..., gt=0, description="Square footage of lot")
    floors: float = Field(..., ge=1, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0 or 1)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Grade rating (1-13)")
    sqft_above: int = Field(..., ge=0, description="Square footage above ground")
    sqft_basement: int = Field(..., ge=0, description="Square footage of basement")
    yr_built: int = Field(..., ge=1800, le=2030, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2030, description="Year renovated (0 if never)")
    zipcode: int = Field(..., description="ZIP code")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    long: float = Field(..., ge=-180, le=180, description="Longitude")
    sqft_living15: int = Field(..., gt=0, description="Living space of nearest 15 neighbors")
    sqft_lot15: int = Field(..., gt=0, description="Lot size of nearest 15 neighbors")
    
    class Config:
        schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2000,
                "sqft_lot": 5000,
                "floors": 2,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 1500,
                "sqft_basement": 500,
                "yr_built": 1990,
                "yr_renovated": 0,
                "zipcode": 98001,
                "lat": 47.5,
                "long": -122.3,
                "sqft_living15": 1800,
                "sqft_lot15": 4500
            }
        }

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted house price")
    prediction_formatted: str = Field(..., description="Formatted price with currency")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version")

# Pydantic model for model info response
class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    train_metrics: dict
    test_metrics: dict
    feature_count: int
    top_features: list
    last_trained: str

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, model_metadata
    
    try:
        # Get paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "models", "rf_model.pkl")
        metrics_path = os.path.join(project_root, "models", "metrics.json")
        
        # Load model
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Load metadata
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metadata = json.load(f)
            print(f"✓ Model metadata loaded from: {metrics_path}")
        else:
            model_metadata = {
                "model": "RandomForestRegressor",
                "timestamp": "Unknown",
                "train_metrics": {},
                "test_metrics": {}
            }
            print("⚠ Model metadata not found, using defaults")
        
        print("✓ API ready to serve predictions")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# Model info endpoint
@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    # Get feature names
    try:
        feature_names = model[:-1].get_feature_names_out()
        feature_count = len(feature_names)
        
        # Get feature importances
        importances = model[-1].feature_importances_
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = [
            {"feature": name, "importance": float(imp)} 
            for name, imp in feature_importance_pairs[:10]
        ]
    except Exception as e:
        feature_count = 0
        top_features = []
    
    return ModelInfo(
        model_name=model_metadata.get("model", "Unknown"),
        model_version="1.0.0",
        train_metrics=model_metadata.get("train_metrics", {}),
        test_metrics=model_metadata.get("test_metrics", {}),
        feature_count=feature_count,
        top_features=top_features,
        last_trained=model_metadata.get("timestamp", "Unknown")
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """Predict house price based on input features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format response
        return PredictionResponse(
            prediction=float(prediction),
            prediction_formatted=f"${prediction:,.2f}",
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)