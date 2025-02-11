from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path
import os
import sys

# Add BiomedParse to path
repo_root = str(Path(__file__).parent.parent.parent)
sys.path.append(repo_root)

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

from .utils.logging_config import setup_logging
from .utils.image_processing import process_dicom, process_nifti, process_rgb
from .utils.monitoring import PREDICTION_TIME, PREDICTION_COUNT
from .schemas import PredictionResponse, PredictionRequest

# Setup logging
logger = setup_logging()

app = FastAPI(title="BiomedParse GPU Server")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info("Initializing BiomedParse model...")
        # Load model configuration
        opt = load_opt_from_config_files([os.path.join(repo_root, "configs/biomedparse_inference.yaml")])
        opt = init_distributed(opt)
        
        # Initialize model
        model = BaseModel(opt, build_model(opt)).from_pretrained('hf_hub:microsoft/BiomedParse').eval().cuda()
        
        # Initialize text embeddings
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], 
                is_eval=True
            )
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

@app.post("/predict", response_model=PredictionResponse)
@PREDICTION_TIME.time()
async def predict(file: UploadFile = File(...), request: PredictionRequest = None):
    PREDICTION_COUNT.inc()
    try:
        # Read file content
        contents = await file.read()
        
        # Determine file type and process accordingly
        if file.filename.lower().endswith(('.dcm', '.dicom')):
            image = await process_dicom(contents)
        elif file.filename.lower().endswith(('.nii', '.nii.gz')):
            image = await process_nifti(contents)
        else:
            image = await process_rgb(contents)
        
        # Process prompts
        prompt_list = request.prompts if request and request.prompts else []
        
        # Run inference
        with torch.no_grad():
            pred_masks = interactive_infer_image(model, image, prompt_list)
        
        # Convert results to list format
        results = []
        for i, pred in enumerate(pred_masks):
            mask_bytes = io.BytesIO()
            mask_img = Image.fromarray((pred * 255).astype(np.uint8))
            mask_img.save(mask_bytes, format='PNG')
            
            results.append({
                'prompt': prompt_list[i],
                'mask': mask_bytes.getvalue(),
                'confidence': float(pred.mean())
            })
        
        return {
            "status": "success",
            "results": results,
            "message": "Prediction completed successfully"
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1) 