from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
from typing import List

# Add BiomedParse to path
repo_root = str(Path(__file__).parent.parent.parent)
sys.path.append(repo_root)

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

from eclypso.srv.utils.logging_config import setup_logging
from eclypso.srv.utils.image_processing import process_dicom, process_nifti, process_rgb
from eclypso.srv.utils.monitoring import PREDICTION_TIME, PREDICTION_COUNT
from eclypso.srv.schemas import PredictionResponse, PredictionRequest, PredictionResult

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
async def predict(
    file: UploadFile = File(...),
    prompts: List[str] = Query(None, description="List of prompts for analysis"),
    request: PredictionRequest = None
):
    PREDICTION_COUNT.inc()
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read file content
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        # Get prompts either from query parameters or request body
        prompt_list = prompts if prompts else (request.prompts if request else [])
        logger.info(f"Using prompts: {prompt_list}")
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No prompts provided")
        
        # Determine file type and process accordingly
        try:
            if file.filename.lower().endswith(('.dcm', '.dicom')):
                logger.info("Processing DICOM file")
                image = await process_dicom(contents)
            elif file.filename.lower().endswith(('.nii', '.nii.gz')):
                logger.info("Processing NIFTI file")
                image = await process_nifti(contents)
            else:
                logger.info("Processing as RGB image")
                image = await process_rgb(contents)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        
        # Run inference
        try:
            pred_masks = []
            with torch.no_grad():
                pred_masks = interactive_infer_image(model, image, prompt_list)
            logger.info("Inference completed successfully")
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        # Convert results to list format
        results = []
        for i, pred in enumerate(pred_masks):
            mask_bytes = io.BytesIO()
            mask_img = Image.fromarray((pred * 255).astype(np.uint8))
            mask_img.save(mask_bytes, format='PNG')
            mask_bytes.seek(0)
            
            # Create PredictionResult instance
            result = PredictionResult(
                prompt=prompt_list[i],
                mask=mask_bytes.getvalue(),
                confidence=float(pred.mean())
            )
            results.append(result)
        
        logger.info(f"Successfully processed {len(results)} predictions")
        
        # Create and return PredictionResponse instance
        return PredictionResponse(
            status="success",
            results=results,
            message="Prediction completed successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
        log_level="debug",  # Enable debug logging
        workers=1
    ) 