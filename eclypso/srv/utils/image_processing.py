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

# Add a helper function to convert DICOM values to JSON-serializable types
def dicom_value_to_serializable(value):
    """Convert DICOM values to JSON-serializable types"""
    if isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    elif isinstance(value, pydicom.multival.MultiValue):
        return [dicom_value_to_serializable(v) for v in value]
    elif hasattr(value, 'original_string'):
        return str(value.original_string)
    elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes, dict)):
        return [dicom_value_to_serializable(v) for v in value]
    return value

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
                axial_aspect = 1.0  # Use isotropic aspect ratio
                slices.append(("Axial", axial_slice, axial_aspect))
                
                # Coronal view (from front)
                coronal_idx = pixel_array.shape[1] // 2
                coronal_slice = pixel_array[:, coronal_idx, :].T  # Transpose to maintain correct orientation
                coronal_aspect = 1.0  # Use isotropic aspect ratio
                slices.append(("Coronal", coronal_slice, coronal_aspect))
                
                # Sagittal view (from side)
                sagittal_idx = pixel_array.shape[0] // 2
                sagittal_slice = pixel_array[sagittal_idx, :, :].T  # Transpose to maintain correct orientation
                sagittal_aspect = 1.0  # Use isotropic aspect ratio
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
                    # Use isotropic aspect ratio
                    axial_aspect = 1.0
                    slices.append(("Axial", axial_slice, axial_aspect))

                    # Coronal view (anterior to posterior)
                    coronal_idx = volume.shape[1] // 2
                    coronal_slice = volume[:, coronal_idx, :]
                    # Flip left-right and up-down to match radiological convention
                    coronal_slice = np.fliplr(np.flipud(coronal_slice))
                    # Use isotropic aspect ratio
                    coronal_aspect = 1.0
                    slices.append(("Coronal", coronal_slice, coronal_aspect))

                    # Sagittal view (left to right)
                    sagittal_idx = volume.shape[0] // 2
                    sagittal_slice = volume[sagittal_idx, :, :]
                    # Flip up-down to match radiological convention
                    sagittal_slice = np.flipud(sagittal_slice)
                    # Use isotropic aspect ratio
                    sagittal_aspect = 1.0
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

async def process_dicomdir(contents_or_path) -> dict:
    """Process DICOMDIR file contents or path and return a structured dictionary of the directory"""
    def _process_dicomdir():
        try:
            logger.info("Starting DICOMDIR processing")
            
            tmp_path = None
            cleanup_tmp = False
            
            # Check if contents_or_path is a string (path) or bytes (file contents)
            if isinstance(contents_or_path, str):
                # It's already a path, use it directly
                tmp_path = contents_or_path
                logger.info(f"Using provided DICOMDIR path: {tmp_path}")
            else:
                # It's bytes content, create a temporary file
                cleanup_tmp = True
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcmdir') as tmp:
                    tmp.write(contents_or_path)
                    tmp_path = tmp.name
                    logger.info(f"Created temporary file: {tmp_path}")
                
            try:
                # Read DICOMDIR data
                dicomdir = pydicom.dcmread(tmp_path)
                
                # Create a structured dictionary to hold the directory information
                directory = {
                    "patients": [],
                    "_path": tmp_path  # Store the path for reference
                }
                
                # Process patient records
                for patient in dicomdir.patient_records:
                    patient_info = {
                        "id": dicom_value_to_serializable(getattr(patient, "PatientID", "Unknown")),
                        "name": dicom_value_to_serializable(getattr(patient, "PatientName", "Unknown")),
                        "studies": []
                    }
                    
                    # Process study records
                    for study in patient.children:
                        if hasattr(study, 'DirectoryRecordType') and study.DirectoryRecordType == 'STUDY':
                            study_info = {
                                "id": dicom_value_to_serializable(getattr(study, "StudyInstanceUID", "Unknown")),
                                "date": dicom_value_to_serializable(getattr(study, "StudyDate", "Unknown")),
                                "description": dicom_value_to_serializable(getattr(study, "StudyDescription", "Unknown Study")),
                                "series": []
                            }
                            
                            # Process series records
                            for series in study.children:
                                if hasattr(series, 'DirectoryRecordType') and series.DirectoryRecordType == 'SERIES':
                                    series_info = {
                                        "id": dicom_value_to_serializable(getattr(series, "SeriesInstanceUID", "Unknown")),
                                        "number": dicom_value_to_serializable(getattr(series, "SeriesNumber", "0")),
                                        "description": dicom_value_to_serializable(getattr(series, "SeriesDescription", "Unknown Series")),
                                        "modality": dicom_value_to_serializable(getattr(series, "Modality", "Unknown")),
                                        "images": []
                                    }
                                    
                                    # Process image records
                                    for image in series.children:
                                        if hasattr(image, 'DirectoryRecordType') and image.DirectoryRecordType == 'IMAGE':
                                            # Get the referenced file path
                                            if hasattr(image, 'ReferencedFileID'):
                                                ref_file = image.ReferencedFileID
                                                # Convert to string if it's a sequence
                                                if isinstance(ref_file, pydicom.multival.MultiValue):
                                                    ref_file = '/'.join(ref_file)
                                                
                                                image_info = {
                                                    "id": dicom_value_to_serializable(getattr(image, "SOPInstanceUID", "Unknown")),
                                                    "number": dicom_value_to_serializable(getattr(image, "InstanceNumber", "0")),
                                                    "path": ref_file
                                                }
                                                series_info["images"].append(image_info)
                                    
                                    # Only add series with images
                                    if series_info["images"]:
                                        study_info["series"].append(series_info)
                            
                            # Only add studies with series
                            if study_info["series"]:
                                patient_info["studies"].append(study_info)
                    
                    # Only add patients with studies
                    if patient_info["studies"]:
                        directory["patients"].append(patient_info)
                
                return directory
                
            finally:
                # Clean up the temporary file only if we created it
                if cleanup_tmp and tmp_path:
                    try:
                        os.unlink(tmp_path)
                        logger.info("Cleaned up temporary file")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing DICOMDIR: {str(e)}", exc_info=True)
            raise
    
    # Run in thread pool
    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, _process_dicomdir)

async def load_dicom_from_dicomdir(dicomdir_path, dicom_path):
    """Load a DICOM file referenced in a DICOMDIR"""
    try:
        # Resolve the DICOM file path relative to the DICOMDIR
        dicomdir_dir = os.path.dirname(dicomdir_path)
        full_dicom_path = os.path.join(dicomdir_dir, dicom_path)
        
        logger.info(f"DICOMDIR directory: {dicomdir_dir}")
        logger.info(f"DICOM relative path: {dicom_path}")
        logger.info(f"Attempting to load from: {full_dicom_path}")
        
        # Check if the file exists
        if not os.path.exists(full_dicom_path):
            # Try alternative path formats (some DICOMDIR use backslashes)
            alt_dicom_path = dicom_path.replace('/', '\\')
            full_dicom_path = os.path.join(dicomdir_dir, alt_dicom_path)
            logger.info(f"First path not found, trying alternative: {full_dicom_path}")
            
            if not os.path.exists(full_dicom_path):
                # Try removing any leading directory part
                base_dicom_path = os.path.basename(dicom_path)
                full_dicom_path = os.path.join(dicomdir_dir, base_dicom_path)
                logger.info(f"Second path not found, trying with basename: {full_dicom_path}")
                
                if not os.path.exists(full_dicom_path):
                    # Try to find the file by its basename in the directory structure
                    logger.info(f"Searching for file with basename: {base_dicom_path}")
                    found_path = find_file_by_name(dicomdir_dir, base_dicom_path)
                    
                    if found_path:
                        full_dicom_path = found_path
                        logger.info(f"Found file at: {full_dicom_path}")
                    else:
                        # Try to find the file by its ID (last part of the path)
                        file_id = os.path.splitext(base_dicom_path)[0]
                        logger.info(f"Searching for file with ID: {file_id}")
                        found_path = find_file_by_pattern(dicomdir_dir, f"*{file_id}*")
                        
                        if found_path:
                            full_dicom_path = found_path
                            logger.info(f"Found file by ID at: {full_dicom_path}")
                        else:
                            # List files in the directory to help diagnose
                            logger.info(f"Files in {dicomdir_dir}:")
                            max_depth = 2
                            for root, dirs, files in os.walk(dicomdir_dir, topdown=True):
                                # Calculate current depth
                                depth = root[len(dicomdir_dir):].count(os.sep)
                                if depth <= max_depth:
                                    rel_path = os.path.relpath(root, dicomdir_dir)
                                    if rel_path == '.':
                                        for file in files[:10]:  # Limit to first 10 files
                                            logger.info(f"  {file}")
                                    else:
                                        logger.info(f"  {rel_path}/ ({len(files)} files)")
                                else:
                                    # Prune directories if we're too deep
                                    dirs[:] = []
                            
                            # Try one more approach - look for DICOM files in the DICOM directory
                            dicom_dir = os.path.join(dicomdir_dir, "DICOM")
                            if os.path.exists(dicom_dir) and os.path.isdir(dicom_dir):
                                logger.info(f"Looking for DICOM files in the DICOM directory")
                                dicom_files = []
                                for root, dirs, files in os.walk(dicom_dir):
                                    for file in files:
                                        if file.upper().startswith('I'):
                                            dicom_files.append(os.path.join(root, file))
                                
                                if dicom_files:
                                    # Use the first DICOM file found
                                    full_dicom_path = dicom_files[0]
                                    logger.info(f"Using first DICOM file found: {full_dicom_path}")
                                else:
                                    logger.error(f"No DICOM files found in the DICOM directory")
                                    return None
                            else:
                                logger.error(f"DICOM file not found: {dicom_path}")
                                return None
        
        # Read the DICOM file
        with open(full_dicom_path, 'rb') as f:
            dicom_contents = f.read()
        
        # Process the DICOM file
        return await process_dicom(dicom_contents)
        
    except Exception as e:
        logger.error(f"Error loading DICOM from DICOMDIR: {str(e)}")
        return None

def find_file_by_name(directory, filename):
    """Find a file by its name in a directory structure"""
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_file_by_pattern(directory, pattern):
    """Find a file matching a pattern in a directory structure"""
    import fnmatch
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            return os.path.join(root, filename)
    return None

async def process_zip_with_dicomdir(contents: bytes) -> dict:
    """Process a ZIP file containing DICOMDIR and associated DICOM files"""
    def _process_zip():
        try:
            logger.info("Starting ZIP with DICOMDIR processing")
            
            # Create a temporary file to save the ZIP contents
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                tmp.write(contents)
                zip_path = tmp.name
                logger.info(f"Created temporary ZIP file: {zip_path}")
            
            # Create a temporary directory to extract the ZIP contents
            extract_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory for extraction: {extract_dir}")
            
            try:
                # Extract the ZIP file
                import zipfile
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # List the contents of the ZIP file for debugging
                        file_list = zip_ref.namelist()
                        logger.info(f"ZIP file contains {len(file_list)} files")
                        if len(file_list) > 0:
                            logger.info(f"First few files: {file_list[:5]}")
                        
                        # Extract all files
                        zip_ref.extractall(extract_dir)
                        logger.info(f"Extracted ZIP file to: {extract_dir}")
                except zipfile.BadZipFile:
                    raise ValueError("The uploaded file is not a valid ZIP file")
                
                # List the extracted contents for debugging
                extracted_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), extract_dir)
                        extracted_files.append(rel_path)
                
                logger.info(f"Extracted {len(extracted_files)} files")
                if len(extracted_files) > 0:
                    logger.info(f"First few extracted files: {extracted_files[:5]}")
                
                # Find the DICOMDIR file
                dicomdir_path = None
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file.upper() == 'DICOMDIR':
                            dicomdir_path = os.path.join(root, file)
                            break
                    if dicomdir_path:
                        break
                
                if not dicomdir_path:
                    raise ValueError("No DICOMDIR file found in the ZIP archive")
                
                logger.info(f"Found DICOMDIR at: {dicomdir_path}")
                
                # Process the DICOMDIR file using the path directly
                # This will be handled by the async function outside this thread
                directory_future = {"_dicomdir_path": dicomdir_path, "_extract_dir": extract_dir}
                
                # Clean up the temporary ZIP file
                try:
                    os.unlink(zip_path)
                    logger.info("Cleaned up temporary ZIP file")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary ZIP file: {str(e)}")
                
                return directory_future
                
            except Exception as e:
                # Clean up the temporary directory in case of error
                try:
                    import shutil
                    shutil.rmtree(extract_dir)
                    logger.info("Cleaned up temporary directory due to error")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary directory: {str(cleanup_error)}")
                raise
                
        except Exception as e:
            logger.error(f"Error processing ZIP with DICOMDIR: {str(e)}", exc_info=True)
            raise
    
    # Run in thread pool to extract the ZIP
    with ThreadPoolExecutor() as executor:
        directory_future = await asyncio.get_event_loop().run_in_executor(executor, _process_zip)
    
    # Now process the DICOMDIR file using its path
    if "_dicomdir_path" in directory_future:
        dicomdir_path = directory_future["_dicomdir_path"]
        extract_dir = directory_future["_extract_dir"]
        
        # Process the DICOMDIR file
        directory = await process_dicomdir(dicomdir_path)
        
        # Add the extract_dir to the directory for reference
        directory["_extract_dir"] = extract_dir
        
        return directory
    else:
        raise ValueError("Failed to find DICOMDIR path in extracted ZIP")

def cleanup_extracted_zip(extract_dir):
    """Clean up the temporary directory created for ZIP extraction"""
    try:
        if extract_dir and os.path.exists(extract_dir):
            import shutil
            shutil.rmtree(extract_dir)
            logger.info(f"Cleaned up temporary directory: {extract_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {str(e)}") 