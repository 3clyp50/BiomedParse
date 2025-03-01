import torch
import numpy as np
from PIL import Image
import cv2
import logging
import asyncio
from typing import List, Dict, Tuple, Any, Optional, Union
import time

logger = logging.getLogger('biomedparse_server')

class ImageCache:
    """
    A class to manage caching of medical images and their processed versions.
    Provides progress tracking and memory usage monitoring.
    """
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize the image cache.
        
        Args:
            max_cache_size: Maximum number of images to cache (0 for unlimited)
        """
        self.max_cache_size = max_cache_size
        self.original_images = {}  # Map of index -> original image
        self.processed_images = {}  # Map of index -> processed image
        self.metadata = {}  # Map of index -> metadata
        self.loading_progress = 0.0
        self.total_images = 0
        self.loaded_images = 0
        self.memory_usage = 0  # Estimated memory usage in bytes
    
    def clear(self):
        """Clear the cache"""
        self.original_images.clear()
        self.processed_images.clear()
        self.metadata.clear()
        self.loading_progress = 0.0
        self.total_images = 0
        self.loaded_images = 0
        self.memory_usage = 0
    
    def is_cached(self, index: int) -> bool:
        """Check if an image is cached"""
        return index in self.original_images
    
    def get_original(self, index: int) -> Optional[Image.Image]:
        """Get an original image from the cache"""
        return self.original_images.get(index)
    
    def get_processed(self, index: int) -> Optional[Image.Image]:
        """Get a processed image from the cache"""
        return self.processed_images.get(index)
    
    def add_image(self, index: int, original: Image.Image, processed: Optional[Image.Image] = None, metadata: Optional[Dict] = None):
        """Add an image to the cache"""
        # Check if we're at capacity
        if self.max_cache_size > 0 and len(self.original_images) >= self.max_cache_size:
            # Remove the oldest image (lowest index)
            oldest_idx = min(self.original_images.keys())
            del self.original_images[oldest_idx]
            if oldest_idx in self.processed_images:
                del self.processed_images[oldest_idx]
            if oldest_idx in self.metadata:
                del self.metadata[oldest_idx]
        
        # Add the new image
        self.original_images[index] = original
        if processed is not None:
            self.processed_images[index] = processed
        if metadata is not None:
            self.metadata[index] = metadata
        
        # Update memory usage estimate
        if original is not None:
            # Estimate memory usage based on image dimensions and bit depth
            width, height = original.size
            channels = len(original.getbands())
            bytes_per_pixel = 1
            if original.mode in ('RGB', 'BGR'):
                bytes_per_pixel = 3
            elif original.mode in ('RGBA', 'CMYK'):
                bytes_per_pixel = 4
            
            image_bytes = width * height * channels * bytes_per_pixel
            self.memory_usage += image_bytes
            
            if processed is not None:
                # Add processed image memory usage
                self.memory_usage += image_bytes
    
    def update_progress(self, loaded: int, total: int):
        """Update loading progress"""
        self.loaded_images = loaded
        self.total_images = total
        if total > 0:
            self.loading_progress = loaded / total
        else:
            self.loading_progress = 1.0
    
    def get_progress(self) -> Tuple[float, int, int]:
        """Get loading progress as (percentage, loaded, total)"""
        return (self.loading_progress, self.loaded_images, self.total_images)
    
    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB"""
        return self.memory_usage / (1024 * 1024)

async def cache_images(
    images: List[Image.Image], 
    model,
    prompts: List[str],
    batch_size: int = 4,
    progress_callback = None
) -> Tuple[List[Image.Image], Dict[str, List[float]]]:
    """
    Cache a list of images with their processed versions.
    
    Args:
        images: List of original images to process
        model: The model to use for inference
        prompts: List of prompts to use for inference
        batch_size: Number of images to process in parallel
        progress_callback: Optional callback function to report progress
        
    Returns:
        Tuple of (list of all images, confidence scores)
    """
    if not images:
        return [], {}
    
    # Initialize results
    all_images = []
    confidences = {prompt: [] for prompt in prompts}
    
    # Process images in batches
    total_images = len(images)
    processed_count = 0
    
    # Create batches
    batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        batch_results = []
        batch_confidences = []
        
        # Process each image in the batch
        for image in batch:
            # Add original image
            all_images.append(image)
            
            # Run inference
            with torch.no_grad():
                pred_masks = await run_inference(model, image, prompts)
            
            # Process predictions
            input_np = np.array(image)
            image_confidences = []
            
            for j, pred in enumerate(pred_masks):
                if pred is None:
                    logger.error(f"Predicted mask {j} is None.")
                    # Add a placeholder for the missing mask
                    all_images.append(image)
                    image_confidences.append(0.0)
                    continue
                
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
                    
                    # Calculate confidence (mean probability in the mask region)
                    confidence = float(pred[binary_mask > 0].mean())
                else:
                    # If mask is empty, just use the original image
                    overlay = input_np.copy()
                    confidence = float(pred.mean())
                
                # Convert overlay to PIL Image
                overlay_img = Image.fromarray(overlay)
                all_images.append(overlay_img)
                image_confidences.append(confidence)
            
            # Store confidences for this image
            for j, prompt in enumerate(prompts):
                if j < len(image_confidences):
                    confidences[prompt].append(image_confidences[j])
                else:
                    confidences[prompt].append(0.0)
        
        # Update progress
        processed_count += len(batch)
        progress = processed_count / total_images
        
        if progress_callback:
            await progress_callback(progress, processed_count, total_images)
        
        # Small delay to allow UI updates
        await asyncio.sleep(0.01)
    
    return all_images, confidences

async def run_inference(model, image, prompts):
    """Run inference on an image with the given prompts"""
    from inference_utils.inference import interactive_infer_image
    
    try:
        return interactive_infer_image(model, image, prompts)
    except Exception as e:
        logger.error(f"Error running inference: {str(e)}")
        return [None] * len(prompts)

def estimate_memory_requirements(images, bit_depth=8, include_processed=True):
    """
    Estimate memory requirements for caching images.
    
    Args:
        images: List of images or number of images
        bit_depth: Bit depth of images (8, 16, 32)
        include_processed: Whether to include processed images in estimate
        
    Returns:
        Estimated memory usage in MB
    """
    if isinstance(images, list):
        num_images = len(images)
        if num_images > 0 and hasattr(images[0], 'size'):
            # Use actual image dimensions
            width, height = images[0].size
        else:
            # Default to typical medical image size
            width, height = 512, 512
    else:
        # Assume images is the number of images
        num_images = images
        width, height = 512, 512
    
    # Calculate bytes per pixel
    bytes_per_pixel = bit_depth / 8
    
    # Calculate memory for one image
    memory_per_image = width * height * bytes_per_pixel
    
    # Calculate total memory
    multiplier = 2 if include_processed else 1
    total_memory_bytes = memory_per_image * num_images * multiplier
    
    # Convert to MB
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    
    return total_memory_mb 