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
import json
import tempfile
import atexit

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
from eclypso.srv.utils.image_processing import (
    process_dicom, process_nifti, process_rgb, 
    process_dicomdir, load_dicom_from_dicomdir,
    process_zip_with_dicomdir, cleanup_extracted_zip,
    dicom_value_to_serializable
)
# Import our new utility modules
from eclypso.srv.utils.image_caching import (
    ImageCache, cache_images, estimate_memory_requirements
)
from eclypso.srv.utils.ui_components import (
    create_image_viewer_components, setup_playback_controls,
    get_javascript_for_image_viewer, get_css_for_image_viewer,
    update_progress_callback, auto_advance
)

# Custom JSON encoder to handle non-serializable objects
class DicomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return dicom_value_to_serializable(obj)
        except:
            return super().default(obj)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biomedparse_gradio")

# Global model instance
model = None
# Global image cache
image_cache = ImageCache(max_cache_size=1000)

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
        if filename.lower() == 'dicomdir':
            # For DICOMDIR, we need a different workflow - return an error message
            return None, "DICOMDIR files should be processed using the DICOMDIR tab"
        elif filename.lower().endswith(('.dcm', '.dicom')):
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

                # Check if mask contains any positive pixels
                if np.any(binary_mask > 0):
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
                else:
                    # If mask is empty, just use the original image
                    logger.warning(f"Mask {j} for image {i} is empty (no values > 0.5). Using original image.")
                    overlay = input_np.copy()

                # Convert overlay to PIL Image
                overlay_img = Image.fromarray(overlay)
                output_images.append(overlay_img)

                # Calculate confidence (mean probability in the mask region)
                if np.any(binary_mask > 0):
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
    
# New function to extract series keys from directory structure
def extract_series_keys(directory):
    """Extract series keys from the directory structure"""
    series_keys = []
    series_mapping = {}  # Map description to key
    try:
        for patient in directory["patients"]:
            patient_id = patient["id"]
            for study in patient["studies"]:
                study_id = study["id"]
                for series in study["series"]:
                    series_id = series["id"]
                    series_key = f"{patient_id}_{study_id}_{series_id}"
                    # Create a descriptive label with index
                    series_desc = f"{series['description']} - {series['modality']} ({len(series['images'])} images)"
                    
                    # Store the mapping
                    series_mapping[series_desc] = series_key
                    
                    # Return both the description and key
                    series_keys.append((series_desc, series_key))
    except Exception as e:
        logger.error(f"Error extracting series keys: {str(e)}")
    return series_keys, series_mapping

# Modified function to process DICOMDIR and return series keys
async def process_dicomdir_file(dicomdir_file):
    try:
        if isinstance(dicomdir_file, str):  # If path is provided
            dicomdir_path = dicomdir_file
            filename = os.path.basename(dicomdir_file)
            logger.info(f"Processing DICOMDIR file from path: {dicomdir_path}")
        else:  # If file object is provided
            contents = dicomdir_file.read()
            filename = dicomdir_file.name
            logger.info(f"Processing uploaded file: {filename}")
            
            # Save to a temporary file to get a path for later reference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcmdir') as tmp:
                tmp.write(contents)
                dicomdir_path = tmp.name
                logger.info(f"Saved uploaded file to temporary path: {dicomdir_path}")

        # Check if it's a ZIP file
        if filename.lower().endswith('.zip'):
            logger.info("Detected ZIP file, processing as ZIP with DICOMDIR")
            if isinstance(dicomdir_file, str):
                # Read the file contents if we only have the path
                with open(dicomdir_path, 'rb') as f:
                    contents = f.read()
            
            directory = await process_zip_with_dicomdir(contents)
            
            # Track the extracted directory for cleanup
            if '_extract_dir' in directory:
                extract_dir = directory['_extract_dir']
                temp_directories.append(extract_dir)
                logger.info(f"Added extracted directory to cleanup list: {extract_dir}")
        else:
            # Process as regular DICOMDIR - pass the path directly
            logger.info(f"Processing as regular DICOMDIR file: {dicomdir_path}")
            directory = await process_dicomdir(dicomdir_path)
        
        # Format the directory structure for display
        formatted_structure = []
        
        for patient in directory["patients"]:
            patient_str = f"Patient: {patient['name']} (ID: {patient['id']})"
            formatted_structure.append(patient_str)
            
            for study in patient["studies"]:
                study_str = f"  Study: {study['description']} (Date: {study['date']})"
                formatted_structure.append(study_str)
                
                for series in study["series"]:
                    series_key = f"{patient['id']}_{study['id']}_{series['id']}"
                    series_str = f"    Series: {series['description']} (Modality: {series['modality']}, Images: {len(series['images'])})"
                    formatted_structure.append(f"{series_str} [Key: {series_key}]")
        
        # Extract series keys for the dropdown
        series_keys, series_mapping = extract_series_keys(directory)
        
        # Store the mapping in the directory structure
        directory["_series_mapping"] = series_mapping
        
        # Save the directory structure to a temporary file for later reference
        directory_path = tempfile.mktemp(suffix='.json')
        with open(directory_path, 'w') as f:
            json.dump(directory, f, cls=DicomJSONEncoder)
        logger.info(f"Saved directory structure to: {directory_path}")
        
        return "\n".join(formatted_structure), directory_path, series_keys
    
    except Exception as e:
        logger.error(f"Error processing DICOMDIR: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", None, []

async def load_series_from_dicomdir(directory_path, series_key, prompts_text):
    try:
        if not directory_path or not os.path.exists(directory_path):
            return None, "Directory information not found"
        
        # Load the directory structure
        with open(directory_path, 'r') as f:
            directory = json.load(f)
        
        # Parse the series key
        try:
            patient_id, study_id, series_id = series_key.split('_')
        except ValueError:
            return None, f"Invalid series key format: {series_key}. Expected format: patient_id_study_id_series_id"
        
        # Find the series
        selected_series = None
        for patient in directory["patients"]:
            if patient["id"] == patient_id:
                for study in patient["studies"]:
                    if study["id"] == study_id:
                        for series in study["series"]:
                            if series["id"] == series_id:
                                selected_series = series
                                break
        
        if not selected_series:
            return None, f"Series not found: {series_key}"
        
        # Get the DICOMDIR path
        dicomdir_path = directory.get("_path")
        if not dicomdir_path:
            return None, "DICOMDIR path not found"
        
        # Check if the DICOMDIR file exists
        if not os.path.exists(dicomdir_path):
            return None, f"DICOMDIR file not found at: {dicomdir_path}"
        
        # Load the middle image from the series
        if not selected_series["images"]:
            return None, "No images found in the selected series"
            
        middle_idx = len(selected_series["images"]) // 2
        middle_image = selected_series["images"][middle_idx]
        
        logger.info(f"Loading image {middle_idx+1} of {len(selected_series['images'])} from series {series_key}")
        logger.info(f"Image path: {middle_image['path']}")
        
        # Try to load the DICOM file using different methods
        images = None
        error_messages = []
        
        # Method 1: Use load_dicom_from_dicomdir
        try:
            logger.info(f"Trying to load DICOM using load_dicom_from_dicomdir")
            images = await load_dicom_from_dicomdir(dicomdir_path, middle_image["path"])
            if images:
                logger.info("Successfully loaded DICOM using load_dicom_from_dicomdir")
        except Exception as e:
            error_message = f"Error loading DICOM using load_dicom_from_dicomdir: {str(e)}"
            logger.warning(error_message)
            error_messages.append(error_message)
        
        # Method 2: If we're dealing with an extracted ZIP, try direct file access
        if not images and "_extract_dir" in directory:
            extract_dir = directory["_extract_dir"]
            logger.info(f"Trying to load DICOM from extracted directory: {extract_dir}")
            
            # Try forward slash path
            dicom_path = os.path.join(extract_dir, middle_image["path"])
            if os.path.exists(dicom_path):
                logger.info(f"Found DICOM file at: {dicom_path}")
                try:
                    with open(dicom_path, 'rb') as f:
                        dicom_contents = f.read()
                    images = await process_dicom(dicom_contents)
                    if images:
                        logger.info("Successfully loaded DICOM from extracted directory (forward slash)")
                except Exception as e:
                    error_message = f"Error processing DICOM from extracted directory (forward slash): {str(e)}"
                    logger.warning(error_message)
                    error_messages.append(error_message)
            else:
                logger.warning(f"DICOM file not found at: {dicom_path}")
                
                # Try backslash path
                alt_dicom_path = middle_image["path"].replace('/', '\\')
                dicom_path = os.path.join(extract_dir, alt_dicom_path)
                if os.path.exists(dicom_path):
                    logger.info(f"Found DICOM file at: {dicom_path} (using backslashes)")
                    try:
                        with open(dicom_path, 'rb') as f:
                            dicom_contents = f.read()
                        images = await process_dicom(dicom_contents)
                        if images:
                            logger.info("Successfully loaded DICOM from extracted directory (backslash)")
                    except Exception as e:
                        error_message = f"Error processing DICOM from extracted directory (backslash): {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
                else:
                    error_message = f"DICOM file not found in extracted directory using either path format: {middle_image['path']}"
                    logger.warning(error_message)
                    error_messages.append(error_message)
        
        if not images:
            error_detail = "\n".join(error_messages) if error_messages else "Unknown error"
            return None, f"Failed to load DICOM image: {middle_image['path']}\n\nDetails:\n{error_detail}"
        
        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split(',')]
        if not prompts or not prompts[0]:
            return None, "Please provide at least one prompt"
        
        # Run inference on the images
        output_images = []
        confidences = []
        view_names = ["Axial", "Coronal", "Sagittal"] if len(images) == 3 else ["Original"]
        
        for i, image in enumerate(images):
            # Add original image
            output_images.append(image)
            
            # Run inference
            with torch.no_grad():
                pred_masks = interactive_infer_image(model, image, prompts)
            
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
                
                # Check if mask contains any positive pixels
                if np.any(binary_mask > 0):
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
                else:
                    # If mask is empty, just use the original image
                    logger.warning(f"Mask {j} for image {i} is empty (no values > 0.5). Using original image.")
                    overlay = input_np.copy()
                
                # Convert overlay to PIL Image
                overlay_img = Image.fromarray(overlay)
                output_images.append(overlay_img)
                
                # Calculate confidence (mean probability in the mask region)
                if np.any(binary_mask > 0):
                    confidence = float(pred[binary_mask > 0].mean())
                else:
                    confidence = float(pred.mean())
                confidences.append(confidence)
        
        # Create result message
        result_message = f"Results for Series: {selected_series['description']}\n"
        for i, (prompt, conf) in enumerate(zip(prompts * len(images), confidences)):
            view_type = view_names[i // len(prompts)]
            result_message += f"- {prompt} ({view_type}): Confidence = {conf:.4f}\n"
        
        return output_images, result_message
    
    except Exception as e:
        logger.error(f"Error loading series from DICOMDIR: {str(e)}")
        return None, f"Error: {str(e)}"

# Create Gradio interface for standard image processing
async def gradio_interface(image_file, prompts_text):
    # Parse prompts
    prompts = [p.strip() for p in prompts_text.split(',')]
    return await process_image(image_file, prompts)

# Create Gradio interface for DICOMDIR processing
async def dicomdir_interface(dicomdir_file):
    structure, directory_path, series_keys = await process_dicomdir_file(dicomdir_file)
    
    # Format series keys for the dropdown - use a list of descriptions only
    dropdown_choices = []
    for desc, key in series_keys:
        dropdown_choices.append(desc)
    
    logger.info(f"Number of dropdown choices: {len(dropdown_choices)}")
    if dropdown_choices:
        logger.info(f"Sample dropdown choices: {dropdown_choices[:3]}")
    else:
        logger.warning("No dropdown choices available!")
    
    # Return the structure, directory path, and dropdown choices
    return structure, directory_path, dropdown_choices

async def series_interface(directory_path, series_desc, prompts_text):
    try:
        # Log the selected description for debugging
        logger.info(f"Series description received: {series_desc}, type: {type(series_desc)}")
        
        if not series_desc:
            return None, "No series selected. Please select a series from the dropdown."
        
        # Load the directory structure to get the mapping
        with open(directory_path, 'r') as f:
            directory = json.load(f)
        
        # Get the mapping from description to key
        series_mapping = directory.get("_series_mapping", {})
        
        # Look up the series key from the description
        series_key = series_mapping.get(series_desc)
        
        if not series_key:
            # If not found in mapping, check if it's already a key
            logger.warning(f"Series description not found in mapping: {series_desc}")
            # Try to use it directly as a key
            series_key = series_desc
        
        logger.info(f"Using series key: {series_key}")
        
        # Parse the series key
        try:
            patient_id, study_id, series_id = series_key.split('_')
        except ValueError:
            return None, f"Invalid series key format: {series_key}. Expected format: patient_id_study_id_series_id"
        
        # Find the series
        selected_series = None
        for patient in directory["patients"]:
            if patient["id"] == patient_id:
                for study in patient["studies"]:
                    if study["id"] == study_id:
                        for series in study["series"]:
                            if series["id"] == series_id:
                                selected_series = series
                                break
        
        if not selected_series:
            return None, f"Series not found: {series_key}"
        
        # Get the DICOMDIR path
        dicomdir_path = directory.get("_path")
        if not dicomdir_path:
            return None, "DICOMDIR path not found"
        
        # Check if the DICOMDIR file exists
        if not os.path.exists(dicomdir_path):
            return None, f"DICOMDIR file not found at: {dicomdir_path}"
        
        # Load all images from the series
        if not selected_series["images"]:
            return None, "No images found in the selected series"
        
        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split(',')]
        if not prompts or not prompts[0]:
            return None, "Please provide at least one prompt"
        
        # Create a progress tracker
        progress = gr.Progress()
        
        # Sort images by instance number if available
        sorted_images = sorted(selected_series["images"], 
                              key=lambda x: int(x["number"]) if isinstance(x["number"], (int, str)) and str(x["number"]).isdigit() else 0)
        
        # First, load all original images
        original_images = []
        total_images = len(sorted_images)
        
        # Estimate memory requirements
        memory_estimate = estimate_memory_requirements(total_images, bit_depth=8, include_processed=True)
        logger.info(f"Estimated memory requirement: {memory_estimate:.2f} MB for {total_images} images")
        
        # Clear the image cache
        global image_cache
        image_cache.clear()
        
        # Load all original images first
        progress(0, f"Loading {len(selected_series['images'])} images from series {selected_series['description']}...")
        
        for idx, image_info in enumerate(sorted_images):
            try:
                logger.info(f"Loading image {idx+1} of {len(sorted_images)} from series {series_key}")
                logger.info(f"Image path: {image_info['path']}")
                
                # Try to load the DICOM file
                images = await load_dicom_from_dicomdir(dicomdir_path, image_info["path"])
                
                if not images:
                    logger.warning(f"Failed to load image {idx+1}, skipping")
                    continue
                
                # For each image loaded (usually just one for 2D DICOM)
                for image in images:
                    original_images.append(image)
                
                # Update progress
                progress((idx + 1) / total_images, f"Loading images: {idx+1}/{total_images}")
                
                # Small delay to allow UI updates
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error loading image {idx+1}: {str(e)}")
                # Continue with next image
        
        if not original_images:
            return None, "Failed to load any images from the series"
        
        # Now process all images with the model
        logger.info(f"Processing {len(original_images)} images with model")
        
        # Define a progress callback function
        async def progress_callback(progress_value, loaded, total):
            progress(progress_value, f"Processing images: {loaded}/{total}")
            await asyncio.sleep(0.01)
        
        # Process all images with the model
        all_images, confidences = await cache_images(
            original_images, 
            model, 
            prompts,
            batch_size=4,  # Process 4 images at a time
            progress_callback=progress_callback
        )
        
        # Create result message
        result_message = f"Results for Series: {selected_series['description']}\n"
        result_message += f"Loaded {len(original_images)} images from the series.\n\n"
        
        # Add average confidence per prompt
        for prompt, conf_values in confidences.items():
            avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0
            result_message += f"- {prompt}: Average Confidence = {avg_confidence:.4f}\n"
        
        # Add memory usage information
        memory_usage = image_cache.get_memory_usage_mb()
        result_message += f"\nMemory usage: {memory_usage:.2f} MB"
        
        return all_images, result_message
    
    except Exception as e:
        logger.error(f"Error in series_interface: {str(e)}", exc_info=True)
        return None, f"Error: {str(e)}"

# Initialize the model at startup
initialize_model()

# Track temporary directories for cleanup
temp_directories = []

def cleanup_temp_directories():
    """Clean up all temporary directories when the application exits"""
    for directory in temp_directories:
        try:
            if directory and os.path.exists(directory):
                import shutil
                shutil.rmtree(directory)
                logger.info(f"Cleaned up temporary directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {str(e)}")

# Register cleanup function
atexit.register(cleanup_temp_directories)

# Create the Gradio interface with tabs
with gr.Blocks(title="BiomedParse Image Analysis", css=get_css_for_image_viewer()) as app:
    gr.Markdown("# BiomedParse Image Analysis")
    
    with gr.Tabs():
        with gr.TabItem("Standard Images"):
            gr.Markdown("""Upload a medical image and provide prompts to analyze specific features.
            For 3D DICOM and NIFTI files, you'll see Axial, Coronal, and Sagittal views with their corresponding predictions.
            For 2D images (including 2D DICOM slices), you'll see the original image and predictions.""")
            
            with gr.Row():
                with gr.Column():
                    standard_file = gr.File(label="Upload Image (DICOM, NIFTI, or standard image formats)")
                    standard_prompts = gr.Textbox(label="Prompts (comma-separated)", placeholder="tumor, lesion")
                    standard_submit = gr.Button("Analyze")
                
                with gr.Column():
                    standard_gallery = gr.Gallery(label="Results", columns=3, show_label=True, height="auto")
                    standard_results = gr.Textbox(label="Confidence Scores", show_label=True)
            
            standard_submit.click(
                fn=gradio_interface,
                inputs=[standard_file, standard_prompts],
                outputs=[standard_gallery, standard_results]
            )
            
            gr.Examples(
                examples=[
                    [os.path.join(examples_dir, "CT_lung_nodule.dcm"), "tumor, nodule"],
                    [os.path.join(examples_dir, "amos_0328.nii.gz"), "worrying area"],
                    [os.path.join(examples_dir, "ISIC_0015551.jpg"), "melanoma, nevus"],
                ],
                inputs=[standard_file, standard_prompts]
            )
        
        with gr.TabItem("DICOMDIR Browser"):
            gr.Markdown("""Upload a DICOMDIR file or a ZIP file containing DICOMDIR to browse its contents and analyze specific series.
            DICOMDIR files are commonly found on medical CDs/DVDs and contain multiple DICOM images organized by patient, study, and series.""")
            
            with gr.Row():
                with gr.Column():
                    dicomdir_file = gr.File(label="Upload DICOMDIR file or ZIP with DICOMDIR")
                    dicomdir_submit = gr.Button("Browse DICOMDIR")
                
                with gr.Column():
                    dicomdir_structure = gr.Textbox(label="DICOMDIR Structure", show_label=True, lines=20)
                    dicomdir_path = gr.Textbox(label="Directory Path", visible=False)
            
            gr.Markdown("## Analyze Series")
            with gr.Row():
                with gr.Column():
                    # Define the dropdown with empty choices initially and allow custom values
                    series_key_dropdown = gr.Dropdown(
                        label="Select Series", 
                        choices=[],
                        value=None,
                        interactive=True,
                        allow_custom_value=True  # Allow typing a custom value as fallback
                    )
                    series_prompts = gr.Textbox(label="Prompts (comma-separated)", placeholder="tumor, lesion")
                    series_submit = gr.Button("Analyze Series")
                
                with gr.Column():
                    # Create image viewer components
                    with gr.Row(elem_classes="player-container"):
                        with gr.Column():
                            # Create the image viewer components
                            viewer_components, viewer_states = create_image_viewer_components()
                            
                            # Extract components for easier access
                            series_slider = viewer_components['slider']
                            prev_button = viewer_components['prev_button']
                            play_button = viewer_components['play_button']
                            stop_button = viewer_components['stop_button']
                            next_button = viewer_components['next_button']
                            speed_slider = viewer_components['speed_slider']
                            loop_checkbox = viewer_components['loop_checkbox']
                            playback_indicator = viewer_components['playback_indicator']
                            image_counter = viewer_components['image_counter']
                            progress_bar = viewer_components['progress_bar']
                            series_image = viewer_components['image']
                            series_results = viewer_components['results']
                            
                            # Extract states for easier access
                            series_images_state = viewer_states['images']
                            playback_state = viewer_states['playback']
                            preload_state = viewer_states['preload']
                            cache_info_state = viewer_states['cache_info']
            
            # Set up playback controls
            setup_playback_controls(viewer_components, viewer_states)
            
            # Add a timer for auto-advancing frames
            # This timer runs every 0.1 seconds and checks if playback is active
            # If playback is active, it advances to the next frame
            # The actual frame rate is controlled by the speed_slider value
            auto_advance_timer = gr.Timer(
                0.1,  # Use a reasonable interval for checking playback status
                auto_advance,  # function to call
                [playback_state, series_slider, series_images_state, speed_slider, loop_checkbox, preload_state],  # inputs
                [playback_state, series_slider, series_image, image_counter, playback_indicator, preload_state]  # outputs
            )
            
            dicomdir_submit.click(
                fn=dicomdir_interface,
                inputs=[dicomdir_file],
                outputs=[dicomdir_structure, dicomdir_path, series_key_dropdown]
            )
            
            # Update the series_submit click handler to use the progress bar
            series_submit.click(
                fn=lambda: {"playing": False, "interval_id": None},  # Reset playback state
                inputs=[],
                outputs=[playback_state]
            ).then(
                fn=lambda: "<div id='preload-trigger' data-action='showPreloading'><span style='color:gray'>Loading...</span></div>",
                inputs=[],
                outputs=[playback_indicator]
            ).then(
                fn=series_interface,
                inputs=[dicomdir_path, series_key_dropdown, series_prompts],
                outputs=[series_images_state, series_results]
            ).then(
                # Add a callback to update the slider and display the first image
                fn=lambda images: (
                    {"minimum": 0, "maximum": len(images) - 1 if images else 0, "value": 0, "step": 1},  # Update slider config with proper maximum
                    [images[0]] if images and len(images) > 0 else [],  # Display first image as a list for gallery
                    f"Image: 1 / {len(images)}" if images else "Image: 0 / 0",  # Update counter
                    {"preloaded": True, "preloaded_indices": [0]}  # Update preload state
                ),
                inputs=[series_images_state],
                outputs=[series_slider, series_image, image_counter, preload_state]
            )

    # Load JavaScript for image viewer
    app.load(js=get_javascript_for_image_viewer() + """
    // Additional initialization for playback
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM fully loaded, initializing playback controls');
        
        // Wait a bit for Gradio to fully initialize
        setTimeout(function() {
            // Find the play button
            const playButtons = document.querySelectorAll('.controls-row button:nth-child(2)');
            if (playButtons.length > 0) {
                console.log(`Found ${playButtons.length} play buttons on page load`);
                
                // Add direct click handlers
                playButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        console.log('Play button clicked on page load');
                        
                        // Get the FPS value from the speed slider
                        const speedSlider = document.querySelector('.speed-slider-container input[type="range"]');
                        const fps = speedSlider ? parseFloat(speedSlider.value) : 10;
                        
                        // Start playback
                        if (window.startPlayback) {
                            window.startPlayback(fps);
                        } else {
                            console.error('startPlayback function not found');
                        }
                    });
                });
            }
            
            // Find the stop button
            const stopButtons = document.querySelectorAll('.controls-row button:nth-child(3)');
            if (stopButtons.length > 0) {
                console.log(`Found ${stopButtons.length} stop buttons on page load`);
                
                // Add direct click handlers
                stopButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        console.log('Stop button clicked on page load');
                        
                        // Stop playback
                        if (window.stopPlayback) {
                            window.stopPlayback();
                        } else {
                            console.error('stopPlayback function not found');
                        }
                    });
                });
            }
        }, 2000);
    });
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0") 