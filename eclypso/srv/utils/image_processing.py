import pydicom
import nibabel as nib
from PIL import Image
import numpy as np
import io
import SimpleITK as sitk
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger('biomedparse_server')

async def process_dicom(contents: bytes) -> Image.Image:
    """Process DICOM file contents and return a PIL Image"""
    def _process_dicom():
        try:
            # Read DICOM data
            dicom_data = pydicom.dcmread(io.BytesIO(contents))
            
            # Get pixel array and apply rescale
            pixel_array = dicom_data.pixel_array.astype(float)
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            
            # Apply window/level adjustment
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
            else:
                # If no window/level in DICOM, estimate them
                window_width = np.max(pixel_array) - np.min(pixel_array)
                window_center = np.min(pixel_array) + (window_width / 2)
            
            # Apply window/level transformation
            min_value = window_center - window_width // 2
            max_value = window_center + window_width // 2
            pixel_array = np.clip(pixel_array, min_value, max_value)
            
            # Normalize to 0-255
            pixel_array = ((pixel_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
            
            # Convert to RGB
            if len(pixel_array.shape) == 2:
                image = Image.fromarray(pixel_array).convert('RGB')
            else:
                image = Image.fromarray(pixel_array[:, :, 0]).convert('RGB')
            
            # Resize to 1024x1024 if needed
            if image.size != (1024, 1024):
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            return image
        
        except Exception as e:
            logger.error(f"Error processing DICOM: {str(e)}")
            raise
    
    # Run in thread pool
    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, _process_dicom)

async def process_nifti(contents: bytes) -> Image.Image:
    """Process NIFTI file contents and return a PIL Image"""
    def _process_nifti():
        try:
            # Load NIFTI data
            nifti_data = nib.load(io.BytesIO(contents))
            volume = nifti_data.get_fdata()
            
            # Take middle slice if 3D
            if len(volume.shape) == 3:
                slice_idx = volume.shape[2] // 2
                slice_data = volume[:, :, slice_idx]
            else:
                slice_data = volume
            
            # Normalize to 0-255
            slice_min = slice_data.min()
            slice_max = slice_data.max()
            if slice_max != slice_min:
                slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                slice_data = np.zeros_like(slice_data, dtype=np.uint8)
            
            # Convert to RGB
            image = Image.fromarray(slice_data).convert('RGB')
            
            # Resize to 1024x1024 if needed
            if image.size != (1024, 1024):
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            return image
        
        except Exception as e:
            logger.error(f"Error processing NIFTI: {str(e)}")
            raise
    
    # Run in thread pool
    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, _process_nifti)

async def process_rgb(contents: bytes) -> Image.Image:
    """Process standard image file contents and return a PIL Image"""
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize to 1024x1024 if needed
        if image.size != (1024, 1024):
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        return image
    
    except Exception as e:
        logger.error(f"Error processing RGB image: {str(e)}")
        raise 