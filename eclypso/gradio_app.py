import gradio as gr
import os
from typing import Tuple, Optional
import os
import shutil
import sys
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from inference_utils.processing_utils import read_rgb

import spaces


MARKDOWN = """
<div align="center" style="padding: 20px 0;">
    <h1 style="font-size: 3em; margin: 0;">
        ሀ<span style="color: #32CD32;">A</span>ኪ<span style="color: #FFD700;">i</span>ም
        <sup style="font-size: 0.5em;">AI</sup>
    </h1>

    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin: 15px 0;">
        <a href="https://cyberbrainai.com/">
            <img src="https://cyberbrainai.com/assets/logo.svg" alt="CyberBrain AI" style="width:40px; height:40px; vertical-align: middle;">
        </a>
        <a href="https://colab.research.google.com/drive/1p3Yf_6xdZPMz5RUtt_NyxrDjrbSgvTDy#scrollTo=t30NqIrCKdAI">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="ድinቅneሽ" style="vertical-align: middle;">
        </a>
        <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y">
            <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="vertical-align: middle;">
        </a>
    </div>
</div>
    <div>
        <p style="font-size: 1.4em; line-height: 1.5; margin: 15px 0; text-align: left;">
            This demo integrates BiomedParse, a foundation model for joint segmentation, detection, and recognition across 9 biomedical imaging modalities. 
            The model supports <span style="color: #FF4500;">CT</span>, <span style="color: #4169E1;">MRI</span>, <span style="color: #32CD32;">X-Ray</span>, <span style="color: #9370DB;">Pathology</span>, <span style="color: #FFD700;">Ultrasound</span>, <span style="color: #FF69B4;">Endoscope</span>, <span style="color: #20B2AA;">Fundus</span>, <span style="color: #FF8C00;">Dermoscopy</span>, and <span style="color: #8B008B;">OCT</span>.
        </p>
    </div>

"""

IMAGE_PROCESSING_EXAMPLES = [
    ["BiomedParse Segmentation", 
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/T0011.jpg",
     "Optic disc in retinal Fundus"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/Part_3_226_pathology_breast.png",
     "optic disc, optic cup"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/covid_1585.png",
     "COVID-19 infection in chest X-Ray"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/TCGA_HT_7856_19950831_8_MRI-FLAIR_brain.png",
     "Lower-grade glioma in brain MRI"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/LIDC-IDRI-0140_143_280_CT_lung.png",
     "COVID-19 infection in chest CT"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/144DME_as_F.jpeg",
     "Cystoid macular edema in retinal OCT"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/Part_1_516_pathology_breast.png",
     "Glandular structure in colon Pathology"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/ISIC_0015551.jpg",
     "Melanoma in skin Dermoscopy"],
    ["BiomedParse Segmentation",
     "https://raw.githubusercontent.com/microsoft/BiomedParse/main/examples/C3_EndoCV2021_00462.jpg",
     "Neoplastic polyp in colon Endoscope"]
]

BIOMEDPARSE_MODES = {
    "CT-Abdomen": ["abdomen", "liver"],
    "CT-Chest": ["lung"],
    "CT-Liver": ["liver"],
    "MRI-Abdomen": ["abdomen"],
    "MRI-Cardiac": ["heart"],
    "MRI-FLAIR-Brain": ["brain"],
    "MRI-T1-Gd-Brain": ["brain"],
    "Pathology": ["bladder", "breast", "cervix", "colon", "esophagus", "kidney", 
                  "liver", "ovarian", "prostate", "stomach", "testis", "thyroid", "uterus"],
    "X-Ray-Chest": ["chest"],
    "Ultrasound-Cardiac": ["heart"],
    "Endoscopy": ["colon"],
    "Fundus": ["retinal"],
    "Dermoscopy": ["skin"],
    "OCT": ["retinal"]
}

IMAGE_INFERENCE_MODES = [
    "BIOMED SEGMENTATION",
    "BIOMED DETECTION", 
    "BIOMED RECOGNITION",
    "BIOMED SEGMENTATION + DETECTION",
    "BIOMED SEGMENTATION + RECOGNITION",
    "BIOMED DETECTION + RECOGNITION",
    "BIOMED SEGMENTATION + DETECTION + RECOGNITION"
]

MODALITY_PROMPTS = {
   "CT-Abdomen": ["postcava", "aorta", "right kidney", "kidney", "left kidney", "duodenum", "pancreas", "liver", "spleen", "stomach", "gallbladder", "left adrenal gland", "adrenal gland", "right adrenal gland", "esophagus"],
   "CT-Chest": ["nodule", "COVID-19 infection", "tumor"],
   "MRI-Abdomen": ["aorta", "postcava", "right kidney", "duodenum", "kidney", "left kidney", "liver", "pancreas", "gallbladder", "stomach", "spleen", "left adrenal gland", "adrenal gland", "right adrenal gland", "esophagus"],
   "MRI-Cardiac": ["left heart ventricle", "myocardium", "right heart ventricle"],
   "MRI-FLAIR-Brain": ["edema", "tumor core", "whole tumor"],
   "MRI-T1-Gd-Brain": ["enhancing tumor", "non-enhancing tumor", "tumor core"],
   "Pathology": ["connective tissue cells", "inflammatory cells", "neoplastic cells", "epithelial cells"],
   "X-Ray-Chest": ["left lung", "lung", "right lung"],
   "Ultrasound-Cardiac": ["left heart atrium", "left heart ventricle"],
   "Endoscopy": ["neoplastic polyp", "polyp", "non-neoplastic polyp"],
   "Fundus": ["optic cup", "optic disc"],
   "Dermoscopy": ["lesion", "melanoma"],
   "OCT": ["edema"]
}

def extract_modality_and_prompts(llm_output):
    """
    Extract modality and relevant prompts from LLM output
    Returns: (modality_type, list_of_prompts)
    """
    llm_output = llm_output.lower()
    
    # Dictionary mapping keywords to modalities
    modality_indicators = {
        'dermatoscop': 'Dermoscopy',
        'dermatoscope': 'Dermoscopy',
        'dermal': 'Dermoscopy',
        'skin lesion': 'Dermoscopy',
        'dermatological': 'Dermoscopy',
        'oct': 'OCT',
        'optical coherence': 'OCT',
        'fundus': 'Fundus',
        'retina': 'Fundus',
        'endoscop': 'Endoscopy',
        'colon': 'Endoscopy',
        'pathological': 'Pathology',
        # 'tissue': 'Pathology',
        'histolog': 'Pathology',
        'x-ray': 'X-Ray-Chest',
        'xray': 'X-Ray-Chest',
        'chest radiograph': 'X-Ray-Chest',
        'mri': None,  # Will be refined below
        'magnetic resonance': None,  # Will be refined below
        'ct': None,  # Will be refined below
        'computed tomography': None,  # Will be refined below
        'ultrasound': 'Ultrasound-Cardiac',
        'sonograph': 'Ultrasound-Cardiac'
    }
    
    # First pass: Detect base modality
    detected_modality = None
    for keyword, modality in modality_indicators.items():
        if keyword in llm_output:
            detected_modality = modality
            break
    
    # Second pass: Refine MRI and CT if detected
    if detected_modality is None and ('mri' in llm_output or 'magnetic resonance' in llm_output):
        if 'brain' in llm_output or 'flair' in llm_output:
            detected_modality = 'MRI-FLAIR-Brain'
        elif 'cardiac' in llm_output or 'heart' in llm_output:
            detected_modality = 'MRI-Cardiac'
        elif 'abdomen' in llm_output:
            detected_modality = 'MRI-Abdomen'
        elif 't1' in llm_output or 'contrast' in llm_output:
            detected_modality = 'MRI-T1-Gd-Brain'
        else:
            detected_modality = 'MRI'
    
    if detected_modality is None and ('ct' in llm_output or 'computed tomography' in llm_output):
        if 'chest' in llm_output or 'lung' in llm_output:
            detected_modality = 'CT-Chest'
        elif 'liver' in llm_output:
            detected_modality = 'CT-Liver'
        elif 'abdomen' in llm_output:
            detected_modality = 'CT-Abdomen'
        else:
            detected_modality = 'CT'
    
    # If still no modality detected, return None
    if not detected_modality:
        return "", []
    
    # Get relevant prompts for the detected modality
    if detected_modality in MODALITY_PROMPTS:
        relevant_prompts = MODALITY_PROMPTS[detected_modality]
    else:
        relevant_prompts = []
    
    return detected_modality, relevant_prompts

def on_mode_dropdown_change(selected_mode):
    if selected_mode in IMAGE_INFERENCE_MODES:
        return [
            gr.Dropdown(visible=True, choices=list(BIOMEDPARSE_MODES.keys()), label="Modality"),
            gr.Dropdown(visible=True, label="Anatomical Site"),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False)
        ]
    else:
        return [
            gr.Dropdown(visible=False),
            gr.Dropdown(visible=False),
            gr.Textbox(visible=True),
            gr.Textbox(visible=(selected_mode == None))
        ]

def on_modality_change(modality):
    if modality:
        return gr.Dropdown(choices=BIOMEDPARSE_MODES[modality], visible=True)
    return gr.Dropdown(visible=False)

def initialize_model():
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    pretrained_pth = 'hf_hub:microsoft/BiomedParse'
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()
    model.to(opt['device'])
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return model

def initialize_llm():
    try:
        print("Starting LLM initialization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModel.from_pretrained(
            "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        print("Model loaded successfully")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
            trust_remote_code=True
        )
        print("Tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to initialize LLM: {str(e)}")
        return None, None

model = initialize_model()
llm_model, llm_tokenizer = initialize_llm()

def update_example_prompts(modality):
    if modality in MODALITY_PROMPTS:
        examples = MODALITY_PROMPTS[modality]
        return f"Example prompts for {modality}:\n" + ", ".join(examples)
    return ""

@spaces.GPU
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(image_path, user_prompt, modality=None):
    try:
        if not image_path:
            return [], "Please upload an image", "No modality detected"
        
        image = read_rgb(image_path)
        pil_image = Image.fromarray(image)
        
        # question = (
        #     f"Analyze this medical image considering the following context: {user_prompt}. "
        #     "Include modality, anatomical structures, and any abnormalities."
        # )
        question = 'What type of medical imaging modality is this?  which organ? Be specific.'
        msgs = [{'role': 'user', 'content': [pil_image, question]}]
        
        llm_response = ""
        if llm_model and llm_tokenizer:
            try:
                for new_text in llm_model.chat(
                    image=pil_image,
                    msgs=msgs,
                    tokenizer=llm_tokenizer,
                    sampling=True,
                    temperature=0.95,
                    stream=True
                ):
                    llm_response += new_text
            except Exception as e:
                print(f"LLM chat error: {str(e)}")
                llm_response = "LLM analysis failed. Proceeding with basic analysis."
        else:
            llm_response = "LLM not available. Please check LLM initialization logs."
        
        detected_modality, relevant_prompts = extract_modality_and_prompts(llm_response)
        if not detected_modality:
            detected_modality = "X-Ray-Chest"  # Fallback modality
            relevant_prompts = MODALITY_PROMPTS["X-Ray-Chest"]
        
        results = []
        analysis_results = []
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        
        # Add color mapping to analysis with more natural language
        color_descriptions = []
        for idx, prompt in enumerate(relevant_prompts):
            color = colors[idx % len(colors)]
            color_name = {(255,0,0): "red", (0,255,0): "green", (0,0,255): "blue", 
                         (255,255,0): "yellow", (255,0,255): "magenta"}[color]
            color_descriptions.append(f"The {prompt} is highlighted in {color_name} color")
        
        for idx, prompt in enumerate(relevant_prompts):
            try:
                mask_list = interactive_infer_image(model, pil_image, [prompt])
                if mask_list is None or len(mask_list) == 0:
                    analysis_results.append(f"No mask generated for '{prompt}'")
                    continue
                
                pred_mask = mask_list[0]
                # Check if mask is valid using numpy's any() function
                if pred_mask is None or not np.any(pred_mask):
                    analysis_results.append(f"Empty mask generated for '{prompt}'")
                    continue
                
                overlay_image = image.copy()
                color = colors[idx % len(colors)]
                mask_indices = pred_mask > 0.5
                if np.any(mask_indices):  # Use np.any() for boolean array check
                    overlay_image[mask_indices] = color
                    results.append(overlay_image)
            except Exception as e:
                print(f"Error processing finding {prompt}: {str(e)}")
                analysis_results.append(f"Failed to process '{prompt}': {str(e)}")
        
        if not results:
            results = [image]  # Return original image if no overlays were created
            
        detailed_analysis = ""
        if llm_model and llm_tokenizer:
            try:
                # Add color legend with more natural language
                # detailed_analysis += "\n\n As shown in the images outputs details:\n \n" + "\n".join(color_descriptions)
                
                analysis_prompt = f"Focus more on the user question. which is: {user_prompt}. Give the modality, organ, analysis, abnormalities (if any), treatment (if abnormalities are present) for this image. "
                msgs = [{'role': 'user', 'content': [pil_image, analysis_prompt]}]
                
                for new_text in llm_model.chat(
                    image=pil_image,
                    msgs=msgs,
                    tokenizer=llm_tokenizer,
                    sampling=True,
                    temperature=0.95,
                    stream=True
                ):
                    detailed_analysis += new_text
                
                # Add segmentation details in a more natural way
                results = []
                analysis_results = []
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
                
                # Add color mapping to analysis with more natural language
                color_descriptions = []
                # First loop: collect prompts found in the analysis
                found_prompts = []
                count = 0
                for idx, prompt in enumerate(relevant_prompts):
                    if prompt in detailed_analysis.lower():
                        color = colors[count % len(colors)]
                        color_name = {(255,0,0): "red", (0,255,0): "green", (0,0,255): "blue", (255,255,0): "yellow", (255,0,255): "magenta"}[color]                        
                        color_descriptions.append(f"The {prompt} is highlighted in {color_name} color for reference")
                        found_prompts.append(prompt)
                        count += 1
                
                # Second loop: only process prompts found in analysis
                for idx, prompt in enumerate(found_prompts):
                    try:
                        mask_list = interactive_infer_image(model, pil_image, [prompt])
                        if mask_list is None or len(mask_list) == 0:
                            analysis_results.append(f"No mask generated for '{prompt}'")
                            continue
                        
                        pred_mask = mask_list[0]
                        # Check if mask is valid using numpy's any() function
                        if pred_mask is None or not np.any(pred_mask):
                            analysis_results.append(f"Empty mask generated for '{prompt}'")
                            continue
                        
                        overlay_image = image.copy()
                        color = colors[idx % len(colors)]
                        mask_indices = pred_mask > 0.5
                        if np.any(mask_indices):  # Use np.any() for boolean array check
                            overlay_image[mask_indices] = color
                            results.append(overlay_image)
                    except Exception as e:
                        print(f"Error processing finding {prompt}: {str(e)}")
                        analysis_results.append(f"Failed to process '{prompt}': {str(e)}")
                
                if not results:
                    results = [image]  # Return original image if no overlays were created
                detailed_analysis += ""
                if color_descriptions:
                    detailed_analysis += " " + " ".join(color_descriptions) + "."
                else:
                    detailed_analysis += " No significant segments were detected."
                
            except Exception as e:
                print(f"LLM chat error: {str(e)}")
                detailed_analysis = "LLM analysis failed. Proceeding with basic analysis."
                if color_descriptions:
                    detailed_analysis += "\n\nHowever, in the segmentation analysis: " + " ".join(color_descriptions) + "."
        else:
            detailed_analysis = "LLM not available. Please check LLM initialization logs."
            if color_descriptions:
                detailed_analysis += "\n\nIn the segmentation analysis: " + " ".join(color_descriptions) + "."
        
        return results, detailed_analysis, detected_modality

    except Exception as e:
        error_msg = f"⚠️ An error occurred: {str(e)}"
        print(f"Error details: {str(e)}", flush=True)
        return [image] if 'image' in locals() else [], error_msg, "Error detecting modality"

with gr.Blocks() as demo:
    gr.HTML(MARKDOWN)    
    with gr.Row():        
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Input Image")
            prompt_input = gr.Textbox(
                lines=2, 
                placeholder="Ask any question about the medical image...", 
                label="Question/Prompt"
            )
            detected_modality = gr.Textbox(
                label="Detected Modality", 
                interactive=False,
                visible=True
            )
            submit_btn = gr.Button("Analyze")
            
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Segmentation Results", 
                show_label=True,
                columns=[2], 
                height="auto"
            )
            analysis_output = gr.Textbox(
                label="Analysis", 
                interactive=False,
                show_label=True,
                lines=10
            )
    
    # Add this to clear outputs when input image is cleared
    image_input.clear(
        lambda: ([], "", ""),
        outputs=[output_gallery, analysis_output, detected_modality]
    )
    
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, prompt_input],
        outputs=[output_gallery, analysis_output, detected_modality],
        api_name="process"
    )

demo.launch()