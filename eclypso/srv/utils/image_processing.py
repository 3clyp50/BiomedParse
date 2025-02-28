import pydicom
import nibabel as nib
from PIL import Image
import numpy as np
import io
import SimpleITK as sitk
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import tempfile
import os

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
            
            # Get pixel spacing
            pixel_spacing = [1.0, 1.0, 1.0]  # Default values
            if hasattr(dicom_data, 'PixelSpacing'):
                pixel_spacing[0:2] = dicom_data.PixelSpacing
            if hasattr(dicom_data, 'SliceThickness'):
                pixel_spacing[2] = dicom_data.SliceThickness
            
            views = []
            # Check if it's a 3D DICOM
            if len(pixel_array.shape) == 3:
                # Prepare slices with correct orientation and aspect ratio
                slices = []
                
                # Axial view (from top)
                axial_idx = pixel_array.shape[2] // 2
                axial_slice = pixel_array[:, :, axial_idx]
                axial_aspect = pixel_spacing[1] / pixel_spacing[0]  # y/x for axial
                slices.append(("Axial", axial_slice, axial_aspect))
                
                # Coronal view (from front)
                coronal_idx = pixel_array.shape[1] // 2
                coronal_slice = pixel_array[:, coronal_idx, :].T  # Transpose to maintain correct orientation
                coronal_aspect = pixel_spacing[2] / pixel_spacing[0]  # z/x for coronal
                slices.append(("Coronal", coronal_slice, coronal_aspect))
                
                # Sagittal view (from side)
                sagittal_idx = pixel_array.shape[0] // 2
                sagittal_slice = pixel_array[sagittal_idx, :, :].T  # Transpose to maintain correct orientation
                sagittal_aspect = pixel_spacing[2] / pixel_spacing[1]  # z/y for sagittal
                slices.append(("Sagittal", sagittal_slice, sagittal_aspect))
                
                for view_name, slice_data, aspect_ratio in slices:
                    # Normalize to 0-255
                    normalized = ((slice_data - min_value) / (max_value - min_value) * 255).astype(np.uint8)
                    
                    # Convert to RGB
                    image = Image.fromarray(normalized).convert('RGB')
                    
                    # Calculate dimensions maintaining aspect ratio
                    target_size = (1024, 1024)
                    # Ensure aspect ratio is a valid number
                    aspect_ratio = float(aspect_ratio)
                    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
                        aspect_ratio = 1.0
                    
                    # Calculate new dimensions
                    if aspect_ratio > 1:
                        new_width = target_size[0]
                        new_height = int(new_width * aspect_ratio)
                        if new_height > target_size[1]:
                            new_height = target_size[1]
                            new_width = int(new_height / aspect_ratio)
                    else:
                        new_height = target_size[1]
                        new_width = int(new_height * aspect_ratio)
                        if new_width > target_size[0]:
                            new_width = target_size[0]
                            new_height = int(new_width / aspect_ratio)
                    
                    # Ensure dimensions are valid
                    new_width = max(1, min(new_width, target_size[0]))
                    new_height = max(1, min(new_height, target_size[1]))
                    
                    # Resize maintaining aspect ratio
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create new image with padding to make it square
                    new_image = Image.new('RGB', target_size, (0, 0, 0))
                    paste_x = (target_size[0] - new_width) // 2
                    paste_y = (target_size[1] - new_height) // 2
                    new_image.paste(image, (paste_x, paste_y))
                    
                    # Add text label
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(new_image)
                    draw.text((10, 10), view_name, fill=(255, 255, 255))
                    
                    views.append(new_image)
            else:
                # Handle 2D case
                normalized = ((pixel_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
                
                # Convert to RGB
                image = Image.fromarray(normalized).convert('RGB')
                
                # Resize to 1024x1024 if needed
                if image.size != (1024, 1024):
                    image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                views = [image]
            
            return views
        
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
            logger.info("Starting NIFTI processing")
            # Create a temporary file to save the bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
                logger.info(f"Created temporary file: {tmp_path}")
            
            try:
                # Load NIFTI data from the temporary file
                logger.info("Loading NIFTI file")
                nifti_img = nib.load(tmp_path)
                if nifti_img is None:
                    raise ValueError("Failed to load NIFTI image")
                
                # Get the data in RAS+ orientation
                logger.info("Converting to canonical orientation")
                nifti_img = nib.as_closest_canonical(nifti_img)
                volume = nifti_img.get_fdata()
                if volume is None:
                    raise ValueError("Failed to get NIFTI data")
                
                logger.info(f"NIFTI volume shape: {volume.shape}")
                logger.info(f"NIFTI data type: {volume.dtype}")
                logger.info(f"NIFTI value range: [{volume.min()}, {volume.max()}]")
                
                # Get pixel dimensions
                pixdim = nifti_img.header.get_zooms()
                logger.info(f"Original pixel dimensions: {pixdim}")
                
                if len(pixdim) < 3:
                    logger.warning("Incomplete pixel dimensions, using default values")
                    pixdim = [1.0, 1.0, 1.0]
                
                # Ensure we have valid positive values
                pixdim = [max(1.0, abs(float(dim))) if dim else 1.0 for dim in pixdim[:3]]
                logger.info(f"Adjusted pixel dimensions: {pixdim}")
                
                views = []
                if len(volume.shape) == 3:
                    logger.info("Processing 3D volume")
                    # Get middle slices for each orientation
                    slices = []
                    
                    # Handle potential NaN or infinite values
                    volume = np.nan_to_num(volume, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
                    
                    # Axial view (superior to inferior)
                    axial_idx = volume.shape[2] // 2
                    axial_slice = volume[:, :, axial_idx]
                    # Flip left-right to match radiological convention
                    axial_slice = np.fliplr(axial_slice)
                    axial_aspect = pixdim[1] / pixdim[0]
                    slices.append(("Axial", axial_slice, axial_aspect))
                    
                    # Coronal view (anterior to posterior)
                    coronal_idx = volume.shape[1] // 2
                    coronal_slice = volume[:, coronal_idx, :]
                    # Flip left-right and up-down to match radiological convention
                    coronal_slice = np.fliplr(np.flipud(coronal_slice))
                    coronal_aspect = pixdim[2] / pixdim[0]
                    slices.append(("Coronal", coronal_slice, coronal_aspect))
                    
                    # Sagittal view (left to right)
                    sagittal_idx = volume.shape[0] // 2
                    sagittal_slice = volume[sagittal_idx, :, :]
                    # Flip up-down to match radiological convention
                    sagittal_slice = np.flipud(sagittal_slice)
                    sagittal_aspect = pixdim[2] / pixdim[1]
                    slices.append(("Sagittal", sagittal_slice, sagittal_aspect))
                    
                    for view_name, slice_data, aspect_ratio in slices:
                        logger.info(f"Processing {view_name} view")
                        # Normalize to 0-255
                        slice_min = slice_data.min()
                        slice_max = slice_data.max()
                        if slice_max != slice_min:
                            normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(slice_data, dtype=np.uint8)
                        
                        # Convert to RGB
                        image = Image.fromarray(normalized).convert('RGB')
                        
                        # Calculate dimensions maintaining aspect ratio
                        target_size = (1024, 1024)
                        # Ensure aspect ratio is a valid number
                        aspect_ratio = float(aspect_ratio)
                        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
                            logger.warning(f"Invalid aspect ratio {aspect_ratio}, using 1.0")
                            aspect_ratio = 1.0
                        
                        logger.info(f"{view_name} aspect ratio: {aspect_ratio}")
                        
                        # Calculate new dimensions
                        if aspect_ratio > 1:
                            new_width = target_size[0]
                            new_height = int(new_width / aspect_ratio)
                            if new_height > target_size[1]:
                                new_height = target_size[1]
                                new_width = int(new_height * aspect_ratio)
                        else:
                            new_height = target_size[1]
                            new_width = int(new_height * aspect_ratio)
                            if new_width > target_size[0]:
                                new_width = target_size[0]
                                new_height = int(new_width / aspect_ratio)
                        
                        # Ensure dimensions are valid
                        new_width = max(1, min(new_width, target_size[0]))
                        new_height = max(1, min(new_height, target_size[1]))
                        
                        logger.info(f"{view_name} dimensions: {new_width}x{new_height}")
                        
                        # Resize maintaining aspect ratio
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Create new image with padding to make it square
                        new_image = Image.new('RGB', target_size, (0, 0, 0))
                        paste_x = (target_size[0] - new_width) // 2
                        paste_y = (target_size[1] - new_height) // 2
                        new_image.paste(image, (paste_x, paste_y))
                        
                        # Add text label
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(new_image)
                        draw.text((10, 10), view_name, fill=(255, 255, 255))
                        
                        views.append(new_image)
                        logger.info(f"Successfully processed {view_name} view")
                else:
                    logger.info("Processing 2D image")
                    # Handle 2D case
                    volume = np.nan_to_num(volume, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
                    slice_min = volume.min()
                    slice_max = volume.max()
                    if slice_max != slice_min:
                        normalized = ((volume - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(volume, dtype=np.uint8)
                    
                    image = Image.fromarray(normalized).convert('RGB')
                    if image.size != (1024, 1024):
                        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                    views = [image]
                    logger.info("Successfully processed 2D image")
                
                if not views:
                    raise ValueError("No views were generated")
                
                return views
            
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                    logger.info("Cleaned up temporary file")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing NIFTI: {str(e)}", exc_info=True)
            raise
    
    # Run in thread pool
    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, _process_nifti)

async def process_rgb(contents: bytes) -> Image.Image:
    """Process standard image file contents and return a PIL Image"""
    def _process_rgb():
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Resize to 1024x1024 if needed
            if image.size != (1024, 1024):
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            return image
        
        except Exception as e:
            logger.error(f"Error processing RGB image: {str(e)}")
            raise
    
    # Run in thread pool
    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, _process_rgb) 