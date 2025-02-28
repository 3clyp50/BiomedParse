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
import cv2

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

async def process_image(image_file, prompts):
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
            images = await process_dicom(contents)  # Returns list of images
        elif filename.lower().endswith(('.nii', '.nii.gz')):
            images = await process_nifti(contents)  # Returns list of images
        else:
            image = await process_rgb(contents)  # Returns single image
            images = [image]  # Convert to list for consistent handling
        
        # Check if images are valid
        if not images or any(img is None for img in images):
            logger.error("No valid images returned from processing.")
            return None, "Error: No valid images returned from processing."
        
        # Run inference on each view
        output_images = []
        confidences = []
        view_names = ["Axial", "Coronal", "Sagittal"] if len(images) == 3 else ["Original"]
        
        for i, image in enumerate(images):
            # Add original image
            output_images.append(image)
            
            # Run inference
            with torch.no_grad():
                pred_masks = interactive_infer_image(model, image, prompts)
                logger.info(f"Predicted masks for image {i}: {pred_masks}")  # Log predicted masks
            
            # Check if pred_masks is valid
            if pred_masks is None or len(pred_masks) == 0:
                logger.error("Predicted masks are None or empty.")
                return None, "Error: Predicted masks are None or empty."
            
            # Convert input image to numpy array
            input_np = np.array(image)
            
            for j, pred in enumerate(pred_masks):
                if pred is None:
                    logger.error(f"Predicted mask {j} is None.")
                    continue  # Skip this mask if it's None
                
                # Create binary mask (threshold at 0.5)
                binary_mask = (pred > 0.5).astype(np.uint8)
                
                # Create colored mask for visualization
                colored_mask = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
                colored_mask[binary_mask > 0] = [255, 0, 0]  # Red for the mask
                
                # Create overlay
                alpha = 0.5
                overlay = input_np.copy()
                overlay[binary_mask > 0] = cv2.addWeighted(
                    overlay[binary_mask > 0], 
                    1-alpha,
                    colored_mask[binary_mask > 0], 
                    alpha, 
                    0
                )
                
                # Convert overlay to PIL Image
                overlay_img = Image.fromarray(overlay)
                output_images.append(overlay_img)
                
                # Calculate confidence (mean probability in the mask region)
                if binary_mask.sum() > 0:
                    confidence = float(pred[binary_mask > 0].mean())
                else:
                    confidence = float(pred.mean())
                confidences.append(confidence)
        
        # Create result message
        result_message = "Results:\n"
        for i, (prompt, conf) in enumerate(zip(prompts * len(images), confidences)):
            view_type = view_names[i // len(prompts)]
            result_message += f"- {prompt} ({view_type}): Confidence = {conf:.4f}\n"
        
        return output_images, result_message
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, f"Error: {str(e)}"

# Create Gradio interface
async def gradio_interface(image_file, prompts_text):
    # Parse prompts
    prompts = [p.strip() for p in prompts_text.split(',')]
    return await process_image(image_file, prompts)

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
        gr.Gallery(label="Results", columns=3, show_label=True, height="auto"),
        gr.Textbox(label="Confidence Scores", show_label=True)
    ],
    title="BiomedParse Image Analysis",
    description="""Upload a medical image and provide prompts to analyze specific features.
    For 3D DICOM and NIFTI files, you'll see Axial, Coronal, and Sagittal views with their corresponding predictions.
    For 2D images (including 2D DICOM slices), you'll see the original image and predictions.
    
    Example prompts:
    - For CT/DICOM: "tumor, lesion, nodule"
    - For Brain MRI/NIFTI: "tumor, edema, necrosis"
    - For Skin lesions: "melanoma, nevus, carcinoma"
    """,
    examples=[
        [os.path.join(examples_dir, "CT_lung_nodule.dcm"), "tumor, nodule"],
        [os.path.join(examples_dir, "amos_0328.nii.gz"), "worrying area"],
        [os.path.join(examples_dir, "ISIC_0015551.jpg"), "melanoma, nevus"],
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0") 