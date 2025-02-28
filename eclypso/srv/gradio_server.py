import gradio as gr
import torch
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import os
import sys
import io
import asyncio  # import here

# Add BiomedParse to path
repo_root = str(Path(__file__).parent.parent.parent)
sys.path.append(repo_root)

# Set examples directory
examples_dir = os.path.join(repo_root, "examples")

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from eclypso.srv.utils.image_processing import process_dicom, process_nifti, process_rgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biomedparse_gradio")

# Global model instance
model = None

def initialize_model():
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
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

def process_image(image_file, prompts):
    try:
        if model is None:
            if not initialize_model():
                return None, "Model initialization failed"
        
        # Read file content
        if isinstance(image_file, str):  # If path is provided
            contents = open(image_file, 'rb').read()
            filename = os.path.basename(image_file)
        else:  # If file object is provided
            contents = image_file.read()
            filename = image_file.name
        
        logger.info(f"Processing file: {filename}")
        
        # Process image based on type
        if filename.lower().endswith(('.dcm', '.dicom')):
            image = asyncio.run(process_dicom(contents))
        elif filename.lower().endswith(('.nii', '.nii.gz')):
            image = asyncio.run(process_nifti(contents))
        else:
            image = process_rgb(contents)
        
        # Run inference
        with torch.no_grad():
            pred_masks = interactive_infer_image(model, image, prompts)
        
        # Convert masks to images
        output_images = []
        confidences = []
        for pred in pred_masks:
            mask_img = Image.fromarray((pred * 255).astype(np.uint8))
            output_images.append(mask_img)
            confidences.append(float(pred.mean()))
        
        # Create result message
        result_message = "Results:\n"
        for prompt, conf in zip(prompts, confidences):
            result_message += f"- {prompt}: Confidence = {conf:.4f}\n"
        
        return output_images, result_message
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, f"Error: {str(e)}"

# Create Gradio interface
def gradio_interface(image_file, prompts_text):
    # Parse prompts
    prompts = [p.strip() for p in prompts_text.split(',')]
    return process_image(image_file, prompts)

# Initialize the model at startup
initialize_model()

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Image (DICOM, NIFTI, or standard image formats)"),
        gr.Textbox(label="Prompts (comma-separated)", placeholder="tumor, lesion")
    ],
    outputs=[
        gr.Gallery(label="Prediction Masks"),
        gr.Textbox(label="Results")
    ],
    title="BiomedParse Image Analysis",
    description="Upload a medical image and provide prompts to analyze specific features.",
    examples=[
        [os.path.join(examples_dir, "CT_lung_nodule.dcm"), "tumor, lesion"],
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0") 