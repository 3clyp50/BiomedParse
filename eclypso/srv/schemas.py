from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    prompts: List[str] = Field(
        default=[],
        description="List of text prompts for analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompts": ["neoplastic cells", "inflammatory cells"]
            }
        }

class PredictionResult(BaseModel):
    """Model for individual prediction result"""
    prompt: str = Field(..., description="The prompt used for this prediction")
    mask: bytes = Field(..., description="Binary mask data in PNG format")
    confidence: float = Field(
        ...,
        description="Confidence score for the prediction",
        ge=0.0,
        le=1.0
    )

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    status: str = Field(..., description="Status of the prediction request")
    results: List[PredictionResult] = Field(
        ...,
        description="List of prediction results"
    )
    message: Optional[str] = Field(
        None,
        description="Additional information about the prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "results": [
                    {
                        "prompt": "neoplastic cells",
                        "mask": "binary_data",
                        "confidence": 0.95
                    }
                ],
                "message": "Prediction completed successfully"
            }
        } 