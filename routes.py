from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from . import model

# Pydantic models for request and response
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: List[int]
    probabilities: List[List[float]]

# Initialize the API router
router = APIRouter()

@router.get("/")
async def health_check():
    """
    Health check endpoint.
    Returns a simple status to indicate the API is running.
    """
    return {"status": "ok"}

@router.post("/api/predict", response_model=PredictionResponse)
async def predict_text(request: PredictionRequest):
    """
    Prediction endpoint.
    Accepts text and returns sentiment prediction and probabilities.
    """
    prediction, probabilities = model.predict(request.text)
    
    if prediction is None or probabilities is None:
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")
    
    return PredictionResponse(prediction=prediction, probabilities=probabilities)
