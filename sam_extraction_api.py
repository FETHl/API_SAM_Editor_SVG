#!/usr/bin/env python3
"""
SAM Extraction API - First part of the two-API architecture
Handles contour extraction and refinement with SAM and CRF
Enhanced memory-optimized version that implements advanced CUDA memory management with FP32 precision
"""

import os
import sys
import uuid
import base64
import json
import logging
import numpy as np
import cv2
import torch
import traceback
import colorsys
import shutil
import psutil
from datetime import datetime, timezone, UTC
from flask import Flask, request, jsonify, send_file, redirect, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sam_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure CUDA for better performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for more stable memory usage
    torch.backends.cudnn.deterministic = True  # More deterministic behavior
    torch.cuda.empty_cache()
    
    # Set memory split size to avoid CUDA OOM errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    # Use a percentage of available GPU memory
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # For older PyTorch versions
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            
    except Exception as e:
        logger.warning(f"Could not set CUDA memory fraction: {e}")

# Import SAM if available - don't load model yet, just check if the module exists
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    HAS_SAM = True
    logger.info("SAM module loaded successfully")
except ImportError:
    logger.warning("SAM module not available. Falling back to basic segmentation.")
    HAS_SAM = False

# Try to import PyDenseCRF for proper edge refinement
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
    HAS_CRF = True
    logger.info("PyDenseCRF loaded successfully")
except ImportError:
    logger.warning("PyDenseCRF not available. Using simplified edge refinement.")
    HAS_CRF = False

# Try to import PIL as a fallback for image saving
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Create Flask app with configuration
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# App configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
    UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', 'uploads'),
    RESULT_FOLDER=os.environ.get('RESULT_FOLDER', 'results'),
    STATIC_FOLDER=os.environ.get('STATIC_FOLDER', 'static'),
    EDITOR_CONTOURS_FOLDER=os.environ.get('EDITOR_CONTOURS_FOLDER', 'contours'),
    EDITOR_UPLOADS_FOLDER=os.environ.get('EDITOR_UPLOADS_FOLDER', 'uploads_editor'),
    SAM_CHECKPOINT=os.environ.get('SAM_CHECKPOINT', './model/sam_vit_h_4b8939.pth'),
    SAM_MODEL_TYPE=os.environ.get('SAM_MODEL_TYPE', 'vit_h'),
    CLEANUP_FILES=os.environ.get('CLEANUP_FILES', 'false').lower() == 'true',
    # Set to False in production!
    DEBUG=os.environ.get('DEBUG', 'false').lower() == 'true',
    # Added config for model unloading
    UNLOAD_MODEL_AFTER_USE=os.environ.get('UNLOAD_MODEL_AFTER_USE', 'true').lower() == 'true',
    # Memory optimization settings
    MAX_IMAGE_SIDE=int(os.environ.get('MAX_IMAGE_SIDE', 1024)),  # Max image dimension for processing
    MAX_IMAGE_PIXELS=int(os.environ.get('MAX_IMAGE_PIXELS', 1000000)),  # Max pixels (width*height)
    # Force FP32 precision - never use half precision
    USE_HALF_PRECISION=False
)

# Create necessary directories
for folder in [
    app.config['UPLOAD_FOLDER'], 
    app.config['RESULT_FOLDER'], 
    app.config['STATIC_FOLDER'], 
    app.config['EDITOR_CONTOURS_FOLDER'],
    app.config['EDITOR_UPLOADS_FOLDER']
]:
    os.makedirs(folder, exist_ok=True)

class SAMExtractor:
    """SAM-based contour extraction with CRF refinement - Memory optimized with FP32 precision"""
    
    def __init__(self, sam_checkpoint=None, model_type="vit_h", device=None):
        """Initialize the SAM Extractor - but don't load model yet"""
        self.sam_model = None
        self.predictor = None
        self.mask_generator = None
        self.sam_masks = None
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        
        # Current date & user - UPDATED with provided values
        self.date = "2025-05-27 10:12:00"  # Updated timestamp
        self.user = "FETHl"  # Updated user login
        
        # Memory tracking
        self.last_image_size = None
        self.fallback_mode = False
        self.use_half_precision = False  # Always false - force FP32 precision
        
        # Set device (CPU or CUDA)
        if device is None:
            self.device = "cpu" if torch.cuda.is_available() else "cuda"
        else:
            self.device = device
            
        # Flag to track if model is loaded
        self.model_loaded = False
        logger.info(f"SAM Extractor initialized (model will be loaded on demand) with FP32 precision")
    
    def is_model_loaded(self):
        """Check if the model is currently loaded"""
        return self.model_loaded and self.sam_model is not None
    
    def monitor_memory(self):
        """Monitor system memory and GPU memory"""
        try:
            # System memory usage
            system_mem = psutil.virtual_memory()
            system_mem_percent = system_mem.percent
            
            # GPU memory usage
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
                total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                gpu_mem_percent = gpu_mem_allocated / total_gpu_mem * 100
                
                logger.debug(f"Memory stats - System: {system_mem_percent:.1f}% used, "
                           f"GPU: {gpu_mem_allocated:.2f}GB/{total_gpu_mem:.2f}GB ({gpu_mem_percent:.1f}% used), "
                           f"Reserved: {gpu_mem_reserved:.2f}GB")
                
                # Return memory status
                return {
                    "system_mem_percent": system_mem_percent,
                    "gpu_mem_allocated": gpu_mem_allocated,
                    "gpu_mem_reserved": gpu_mem_reserved,
                    "total_gpu_mem": total_gpu_mem,
                    "gpu_mem_percent": gpu_mem_percent
                }
            else:
                logger.debug(f"Memory stats - System: {system_mem_percent:.1f}% used, GPU: N/A")
                return {
                    "system_mem_percent": system_mem_percent,
                    "gpu_mem_allocated": 0,
                    "gpu_mem_reserved": 0,
                    "total_gpu_mem": 0,
                    "gpu_mem_percent": 0
                }
        except Exception as e:
            logger.warning(f"Error monitoring memory: {e}")
            return {}
            
    def estimate_max_size(self):
        """Estimate the maximum image size we can process based on available memory"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                # Get total GPU memory
                total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Get currently used memory
                allocated_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                
                # Calculate free memory with a safety margin
                free_mem = total_gpu_mem - allocated_mem - 0.5  # Leave 0.5GB safety margin
                
                # Adjusted values for FP32 precision on different memory sizes
                if total_gpu_mem < 6.0:  # For GPUs with less than 6GB VRAM
                    if free_mem <= 1.0:
                        return 300000  # ~550x550px
                    elif free_mem <= 2.0:
                        return 500000  # ~700x700px  
                    else:
                        return 650000  # ~800x800px
                else:  # For GPUs with 6GB+ VRAM
                    if free_mem <= 2.0:
                        return 600000  # ~775x775px
                    elif free_mem <= 4.0:
                        return 800000  # ~895x895px
                    elif free_mem <= 8.0:
                        return 1200000  # ~1095x1095px
                    else:
                        return 1600000  # ~1265x1265px
            else:
                return app.config['MAX_IMAGE_PIXELS']  # Default value
        except Exception as e:
            logger.warning(f"Error estimating max size: {e}")
            return app.config['MAX_IMAGE_PIXELS']  # Default value
    
    def load_sam_model(self):
        """Load the SAM model from checkpoint - only when needed, using FP32 precision"""
        # If model already loaded, do nothing
        if self.is_model_loaded():
            logger.info("SAM model already loaded, skipping load")
            return True
            
        try:
            if not HAS_SAM:
                logger.warning("SAM module not available, cannot load model")
                return False
                
            # Check if checkpoint exists
            checkpoint_path = self.sam_checkpoint
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                # Try to find SAM checkpoint in the current directory
                checkpoint_path = app.config['SAM_CHECKPOINT']
                if not os.path.exists(checkpoint_path):
                    logger.error(f"SAM checkpoint not found: {checkpoint_path}")
                    return False
            
            logger.info(f"Loading SAM model from {checkpoint_path} with FP32 precision...")
            
            # Clear cache before loading to ensure maximum available memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load the model with explicit FP32 precision
            # Use the newer recommended syntax for autocast
            with torch.amp.autocast('cuda', enabled=False):
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
                sam.to(device=self.device)
                
                # Ensure model is in full precision
                sam = sam.float()
                logger.info("Using full precision (FP32) for better accuracy")
                
                # Verify model is using FP32
                for param in sam.parameters():
                    if param.dtype != torch.float32:
                        param.data = param.data.float()
            
            self.sam_model = sam
            self.predictor = SamPredictor(sam)
            
            # Initialize mask generator with memory-efficient settings
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=16,         # Reduced from 32 to save memory
                points_per_batch=64,        # Reduce batch size to save memory
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,            # Reduced cropping to save memory
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            # Set the flag to indicate model is loaded
            self.model_loaded = True
            
            # Reset fallback mode
            self.fallback_mode = False
            
            logger.info(f"SAM model loaded successfully on {self.device} with FP32 precision")
            return True
        except Exception as e:
            logger.error(f"Error loading SAM model: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Ensure flag is set to false in case of error
            self.model_loaded = False
            return False
    
    def unload_model(self):
        """Unload the model and free GPU memory"""
        if not self.is_model_loaded():
            return  # Nothing to do
            
        logger.info("Unloading SAM model to free GPU memory")
        
        try:
            # Remove references to models
            self.sam_model = None
            self.predictor = None
            self.mask_generator = None
            
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("SAM model unloaded successfully")
            
            # Update flag
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
    
    def process_image_for_sam(self, image):
        """
        Preprocess image for SAM processing with memory-efficient resizing
        
        Parameters:
        - image: BGR image from OpenCV
        
        Returns:
        - Processed RGB image, scale factor, target dimensions
        """
        height, width = image.shape[:2]
        original_size = (width, height)
        total_pixels = width * height
        
        # Convert to RGB for SAM
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get current memory stats
        mem_stats = self.monitor_memory()
        
        # Calculate max size based on current memory
        max_pixels = self.estimate_max_size()
        logger.info(f"Estimated max image size: {max_pixels} pixels")
        
        # Determine if we need to resize
        if total_pixels > max_pixels:
            # Calculate scale factor to fit within max_pixels
            scale = np.sqrt(max_pixels / total_pixels)
            
            # Calculate target dimensions
            target_width = int(width * scale)
            target_height = int(height * scale)
            
            logger.info(f"Resizing image from {width}x{height} to {target_width}x{target_height} for processing")
            
            # Resize image for processing
            processed_image = cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            return processed_image, scale, (target_width, target_height)
        else:
            # No resizing needed
            return rgb_image, 1.0, original_size
    
    def segment_automatic(self, image):
        """
        Perform automatic segmentation without prompts
        
        Parameters:
        - image: BGR image (OpenCV format)
        
        Returns:
        - Binary mask as boolean array
        """
        # Load model if not already loaded
        if not self.is_model_loaded():
            if not self.load_sam_model():
                # If model can't be loaded, fall back to basic segmentation
                logger.warning("SAM model could not be loaded. Using basic thresholding.")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return mask > 0
        
        try:
            # Check if image is very large
            height, width = image.shape[:2]
            original_size = (width, height)
            
            # Process and resize image for SAM
            processed_image, scale_factor, (target_width, target_height) = self.process_image_for_sam(image)
            
            # Try to use automatic mask generator
            try:
                # Clear GPU cache before processing
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Disable any automatic mixed precision during processing
                with torch.amp.autocast('cuda', enabled=False):
                    with torch.no_grad():  # Ensure no gradients tracked for memory efficiency
                        masks = self.mask_generator.generate(processed_image)
                    
                logger.info(f"SAM generated {len(masks)} masks")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"SAM mask generation failed: {e}")
                    # Set fallback mode for next time
                    self.fallback_mode = True
                    
                    # Use a basic fallback method
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Create a circle in the center as a default mask
                    center_mask = np.zeros((height, width), dtype=np.uint8)
                    center_x, center_y = width // 2, height // 2
                    radius = min(width, height) // 4
                    cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
                    
                    # Unload model after processing if configured
                    if app.config['UNLOAD_MODEL_AFTER_USE']:
                        self.unload_model()
                    
                    return center_mask > 0
                else:
                    # Re-raise other errors
                    raise
            
            # Initialize combined mask and individual masks storage
            if scale_factor < 1.0:  # If we resized earlier
                combined_mask = np.zeros((target_height, target_width), dtype=np.uint8)
                self.sam_masks = []
                
                # Process individual masks
                for i, mask_data in enumerate(masks):
                    mask = mask_data["segmentation"].astype(np.uint8) * 255
                    
                    # Store for later processing (resized to original)
                    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    self.sam_masks.append({
                        "segmentation": resized_mask > 127,
                        "score": mask_data.get("score", 1.0),
                        "area": float(np.sum(resized_mask > 0))
                    })
                    
                    # Update combined mask
                    combined_mask = np.maximum(combined_mask, mask)
                
                # Resize combined mask back to original size
                final_mask = cv2.resize(combined_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                # Process for original-sized images
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                self.sam_masks = []
                
                for i, mask_data in enumerate(masks):
                    mask = mask_data["segmentation"].astype(np.uint8) * 255
                    
                    # Store for later processing
                    self.sam_masks.append({
                        "segmentation": mask > 127,
                        "score": mask_data.get("score", 1.0),
                        "area": float(np.sum(mask > 0))
                    })
                    
                    # Update combined mask
                    combined_mask = np.maximum(combined_mask, mask)
                
                final_mask = combined_mask
            
            # Make sure we have a binary mask
            binary_mask = final_mask > 127
            
            # Unload model after processing if configured
            if app.config['UNLOAD_MODEL_AFTER_USE']:
                self.unload_model()
            
            # Return the mask as binary
            return binary_mask
            
        except Exception as e:
            logger.error(f"Error in automatic segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create a circle in the center as a default mask
            center_mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
            
            # Make sure to unload model even in case of error
            if app.config['UNLOAD_MODEL_AFTER_USE']:
                self.unload_model()
            
            return center_mask > 0
    
    def segment_with_points(self, image, points, labels=None):
        """
        Segment image using point prompts
        
        Parameters:
        - image: BGR image (OpenCV format)
        - points: Nx2 array of (x, y) point coordinates
        - labels: N-length array of 1 (foreground) or 0 (background) (default: all 1)
        
        Returns:
        - Binary mask as boolean array
        """
        # Load model if not already loaded
        if not self.is_model_loaded():
            logger.warning("SAM predictor not available for point prompts. Using automatic segmentation.")
            return self.segment_automatic(image)
        
        try:
            # Check image dimensions
            height, width = image.shape[:2]
            original_size = (width, height)
            
            # Set default labels if not provided (all foreground)
            if labels is None:
                labels = np.ones(len(points))
                
            # Convert to numpy arrays if not already
            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(labels, dtype=np.int32)
            
            # Process and resize image for SAM
            processed_image, scale_factor, (target_width, target_height) = self.process_image_for_sam(image)
            
            # Scale points to match the resized image if needed
            if scale_factor != 1.0:
                scaled_points = input_points * scale_factor
            else:
                scaled_points = input_points
            
            try:
                # Set the image in SAM with explicit FP32 precision
                with torch.amp.autocast('cuda', enabled=False):  # Ensure FP32 precision
                    self.predictor.set_image(processed_image)
                
                # Free memory before prediction
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Get the mask prediction with explicit FP32 precision
                with torch.amp.autocast('cuda', enabled=False):  # Ensure FP32 precision
                    with torch.no_grad():  # Ensure no gradients tracked for memory efficiency
                        masks, scores, _ = self.predictor.predict(
                            point_coords=scaled_points,
                            point_labels=input_labels,
                            multimask_output=True  # Get multiple mask predictions
                        )
            except Exception as e:
                logger.error(f"Error in point segmentation: {e}")
                # Fall back to automatic segmentation
                return self.segment_automatic(image)
            
            # Store masks for contour extraction
            self.sam_masks = []
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                
                for i, mask in enumerate(masks):
                    if scale_factor != 1.0:
                        # Need to resize mask to original dimensions
                        resized_mask = cv2.resize((mask > 0).astype(np.uint8) * 255, 
                                        original_size, interpolation=cv2.INTER_NEAREST)
                        mask_for_segmentation = resized_mask > 127
                    else:
                        mask_for_segmentation = mask > 0
                        resized_mask = (mask > 0).astype(np.uint8) * 255
                    
                    self.sam_masks.append({
                        "segmentation": mask_for_segmentation,
                        "score": float(scores[i]),
                        "area": float(np.sum(mask_for_segmentation))
                    })
                
                # Return the best mask resized to original dimensions if needed
                if scale_factor != 1.0:
                    best_mask = cv2.resize((masks[best_mask_idx] > 0).astype(np.uint8) * 255, 
                                    original_size, interpolation=cv2.INTER_NEAREST)
                    result = best_mask > 127
                else:
                    result = masks[best_mask_idx] > 0
                
                # Unload model after processing if configured
                if app.config['UNLOAD_MODEL_AFTER_USE']:
                    self.unload_model()
                    
                return result
            else:
                logger.warning("No masks generated from points")
                # Fall back to automatic segmentation
                return self.segment_automatic(image)
                
        except Exception as e:
            logger.error(f"Error in point segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Always unload model at the end if configured
            if app.config['UNLOAD_MODEL_AFTER_USE']:
                self.unload_model()
                
            # Fall back to automatic segmentation
            return self.segment_automatic(image)
    
    def _create_default_mask(self, image, points=None):
        """Create a default mask when segmentation fails"""
        height, width = image.shape[:2]
        
        # If we have points, create a mask based on those points
        if points is not None and len(points) > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y in points:
                cv2.circle(mask, (int(x), int(y)), 50, 255, -1)
            return mask > 0
        else:
            # Otherwise create a center circle
            center_mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
            return center_mask > 0
    
    def segment_with_box(self, image, box):
        """
        Segment image using a bounding box prompt
        
        Parameters:
        - image: BGR image (OpenCV format)
        - box: 2x2 array [[x1, y1], [x2, y2]] of box corners
        
        Returns:
        - Binary mask as boolean array
        """
        # Load model if not already loaded
        if not self.is_model_loaded():
            logger.warning("SAM predictor not available for box prompts. Using automatic segmentation.")
            return self.segment_automatic(image)
        
        try:
            # Check image dimensions
            height, width = image.shape[:2]
            original_size = (width, height)
            
            # Convert box to the format SAM expects [x1, y1, x2, y2]
            input_box = np.array([box[0][0], box[0][1], box[1][0], box[1][1]], dtype=np.float32)
            
            # Process and resize image for SAM
            processed_image, scale_factor, (target_width, target_height) = self.process_image_for_sam(image)
            
            # Scale box to match the resized image if needed
            if scale_factor != 1.0:
                scaled_box = input_box * scale_factor
            else:
                scaled_box = input_box
            
            try:
                # Set the image in SAM with explicit FP32 precision
                with torch.amp.autocast('cuda', enabled=False):
                    self.predictor.set_image(processed_image)
                
                # Free memory before prediction
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Get the mask prediction with explicit FP32 precision
                with torch.amp.autocast('cuda', enabled=False):
                    with torch.no_grad():  # Ensure no gradients tracked for memory efficiency
                        masks, scores, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=scaled_box[None, :],
                            multimask_output=True  # Get multiple mask predictions
                        )
            except Exception as e:
                logger.error(f"Error in box segmentation: {e}")
                # Fall back to automatic segmentation
                return self.segment_automatic(image)
            
            # Store masks for contour extraction
            self.sam_masks = []
            
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                
                for i, mask in enumerate(masks):
                    if scale_factor != 1.0:
                        # Need to resize mask to original dimensions
                        resized_mask = cv2.resize((mask > 0).astype(np.uint8) * 255, 
                                        original_size, interpolation=cv2.INTER_NEAREST)
                        mask_for_segmentation = resized_mask > 127
                    else:
                        mask_for_segmentation = mask > 0
                        resized_mask = (mask > 0).astype(np.uint8) * 255
                    
                    self.sam_masks.append({
                        "segmentation": mask_for_segmentation,
                        "score": float(scores[i]),
                        "area": float(np.sum(mask_for_segmentation))
                    })
                
                # Return the best mask resized to original dimensions
                if scale_factor != 1.0:
                    best_mask = cv2.resize((masks[best_mask_idx] > 0).astype(np.uint8) * 255, 
                                    original_size, interpolation=cv2.INTER_NEAREST)
                    result = best_mask > 127
                else:
                    result = masks[best_mask_idx] > 0
                
                # Unload model after processing if configured
                if app.config['UNLOAD_MODEL_AFTER_USE']:
                    self.unload_model()
                    
                return result
            else:
                logger.warning("No masks generated from box")
                # Fall back to automatic segmentation
                return self.segment_automatic(image)
                
        except Exception as e:
            logger.error(f"Error in box segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Always unload model at the end if there's an error
            if app.config['UNLOAD_MODEL_AFTER_USE']:
                self.unload_model()
                
            # Fall back to automatic segmentation
            return self.segment_automatic(image)
            
    def refine_edges(self, image, mask):
        """
        Refine the edges of a mask using CRF (if available) or morphological operations
        
        Parameters:
        - image: Original RGB image
        - mask: Binary segmentation mask
        
        Returns:
        - Refined binary mask
        """
        # Ensure mask is correct format - 2D binary mask
        if len(mask.shape) > 2:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                # Take first channel or combine channels
                mask = np.any(mask, axis=2)
        
        # Convert mask to uint8 binary format (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Apply CRF if available - with error handling
        if HAS_CRF:
            try:
                return self._apply_crf(image, binary_mask)
            except Exception as e:
                logger.warning(f"CRF application error: {str(e)}")
                logger.warning(traceback.format_exc())
                # Return original mask on failure
                return mask > 0
        
        # Morphological refinement
        try:
            # Ensure mask has correct shape
            if binary_mask.shape[:2] != image.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
                
            # Use a series of morphological operations for refinement
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_medium = np.ones((5, 5), np.uint8)
            
            # Close small gaps
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_small)
            
            # Remove small noise
            opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
            
            # Smooth edges
            smooth = cv2.GaussianBlur(opening, (5, 5), 0)
            refined_mask = smooth > 0.5
            
            return refined_mask
        except Exception as e:
            logger.error(f"Edge refinement error: {str(e)}")
            logger.error(traceback.format_exc())
            return mask
    
    def _apply_crf(self, image, mask, iterations=5):
        """
        Apply Conditional Random Field to refine mask edges
        
        Parameters:
        - image: RGB image
        - mask: Binary mask (0 or 1)
        - iterations: Number of CRF iterations
        
        Returns:
        - Refined binary mask
        """
        try:
            # Ensure shapes match
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
            h, w = mask.shape[:2]
            
            # FIX: Make absolutely sure the mask only contains 0 and 1 values
            # This fixes the "index 255 is out of bounds for axis 0 with size 2" error
            mask_labels = np.zeros_like(mask)
            mask_labels[mask > 0] = 1
            
            # Double-check unique values
            unique_values = np.unique(mask_labels)
            logger.info(f"Mask labels min: {mask_labels.min()}, max: {mask_labels.max()}, unique: {unique_values}")
            
            # Create CRF with the image dimensions
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 labels: fg and bg
            
            # Create unary potentials from the mask
            U = unary_from_labels(mask_labels, 2, gt_prob=0.7)
            d.setUnaryEnergy(U)
            
            # Create pairwise potentials (bilateral)
            # This considers both color similarity and proximity
            pairwise_energy = create_pairwise_bilateral(
                sdims=(80, 80),  # Spatial dimensions (sigma_x, sigma_y)
                schan=(13, 13, 13),  # Color dimensions (sigma_r, sigma_g, sigma_b)
                img=image,
                chdim=2  # Color is in the 3rd dimension (RGB)
            )
            d.addPairwiseEnergy(pairwise_energy, compat=10)
            
            # Add another pairwise term (Potts model)
            # This penalizes small isolated regions
            d.addPairwiseEnergy(
                np.ones(w*h, dtype=np.float32),  # Flattened array of correct size
                compat=3,
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )
            
            # Perform inference
            Q = d.inference(iterations)
            
            # Get the refined mask
            refined_mask = np.argmax(Q, axis=0).reshape((h, w)) * 255
            
            return refined_mask > 0
            
        except Exception as e:
            logger.warning(f"CRF application error: {str(e)}")
            logger.warning(traceback.format_exc())
            # Return original mask on failure
            return mask > 0

    def extract_contours(self, mask):
        """
        Extract contours from binary mask and SAM masks if available
        
        Parameters:
        - mask: Combined binary mask
        
        Returns:
        - List of contours
        """
        # Store results
        all_contours = []
        contour_id = 0
        
        # Debug info
        logger.info(f"Extracting contours from mask: shape={mask.shape}, " 
                   f"min={np.min(mask)}, max={np.max(mask)}")
        
        # Create debug image for visualization
        height, width = mask.shape[:2]
        debug_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Ensure mask is binary and not empty
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Check if mask is empty
        if np.sum(binary_mask) == 0:
            logger.warning("Empty mask detected, creating default contours")
            
            # Create several contours across the image for better results
            contours_to_add = []
            
            # Add a rectangle covering ~60% of the image
            rect_w, rect_h = int(width * 0.6), int(height * 0.6)
            x1 = (width - rect_w) // 2
            y1 = (height - rect_h) // 2
            x2, y2 = x1 + rect_w, y1 + rect_h
            contours_to_add.append({
                'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                'color': [0.8, 0.2, 0.2],  # Red
                'area': float(rect_w * rect_h)
            })
            
            # Add a circle in the center
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 5
            circle_points = []
            for angle in range(0, 360, 10):  # Every 10 degrees
                x = center_x + int(radius * np.cos(np.radians(angle)))
                y = center_y + int(radius * np.sin(np.radians(angle)))
                circle_points.append([x, y])
            contours_to_add.append({
                'points': circle_points,
                'color': [0.2, 0.8, 0.2],  # Green
                'area': float(np.pi * radius * radius)
            })
            
            # Add contours to the results
            for i, contour_data in enumerate(contours_to_add):
                all_contours.append({
                    'id': i,
                    'mask_id': -1,
                    'points': contour_data['points'],
                    'is_external': True,
                    'parent_idx': -1,
                    'color': contour_data['color'],
                    'area': contour_data['area']
                })
            
            # Save a debug image to help diagnose the issue
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(debug_image, (center_x, center_y), radius, (0, 255, 0), 2)
            debug_path = os.path.join(app.config['RESULT_FOLDER'], f"contour_debug_{uuid.uuid4().hex[:8]}.png")
            cv2.imwrite(debug_path, debug_image)
            logger.info(f"Saved contour debug image to {debug_path}")
            
            return all_contours
        
        # 1. First, try to extract contours from individual SAM masks
        if hasattr(self, 'sam_masks') and self.sam_masks and len(self.sam_masks) > 0:
            logger.info(f"Using {len(self.sam_masks)} individual SAM masks for contour extraction")
            
            for idx, mask_data in enumerate(self.sam_masks):
                # Get the individual mask
                individual_mask = mask_data["segmentation"].astype(np.uint8) * 255
                
                # Make sure the mask is not empty
                if np.sum(individual_mask) == 0:
                    continue
                
                # Generate contour color
                hue = (idx * 137.5) % 360  # Use golden angle for good distribution
                r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                color = [float(r), float(g), float(b)]
                cv_color = (int(r*255), int(g*255), int(b*255))
                
                # Find contours in this mask
                try:
                    # Use RETR_EXTERNAL first for cleaner boundaries
                    contours, _ = cv2.findContours(
                        individual_mask, 
                        cv2.RETR_EXTERNAL,  
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if not contours:
                        # Try RETR_CCOMP if no external contours
                        contours, hierarchy = cv2.findContours(
                            individual_mask, 
                            cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        hierarchy = hierarchy[0] if hierarchy is not None else []
                    else:
                        # Simple hierarchy for external contours
                        hierarchy = [[-1, -1, -1, -1] for _ in range(len(contours))]
                    
                    logger.info(f"Found {len(contours)} contours in mask {idx}")
                    
                    # Process each contour
                    for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                        # Filter very small contours
                        area = cv2.contourArea(contour)
                        if area < 15:  # Smaller threshold to catch more contours
                            continue
                            
                        # Check if contour has enough points
                        if len(contour) < 3:
                            continue
                        
                        # Simplify contour slightly
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Convert to points list
                        points = approx.reshape(-1, 2).tolist()
                        
                        # Determine external vs hole
                        is_external = True if len(hierarchy) == 0 else (h[3] == -1)
                        parent_idx = h[3] if not is_external else -1
                        
                        # Add to results
                        all_contours.append({
                            'id': contour_id,
                            'mask_id': idx,
                            'points': points,
                            'is_external': bool(is_external),
                            'parent_idx': int(parent_idx),
                            'color': color,
                            'area': float(area)
                        })
                        
                        # For debugging
                        cv2.drawContours(debug_image, [approx], -1, cv_color, 1)
                        
                        contour_id += 1
                        
                except Exception as e:
                    logger.error(f"Error extracting contours from mask {idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue
        
        # 2. If no contours from individual masks, try with the combined mask
        if len(all_contours) == 0:
            logger.info("Falling back to combined mask contour extraction")
            
            # Make sure the mask has full range for good contour detection
            normalized_mask = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX)
            
            # Try different contour retrieval modes
            contour_found = False
            for mode in [cv2.RETR_EXTERNAL, cv2.RETR_CCOMP, cv2.RETR_TREE]:
                try:
                    contours, hierarchy = cv2.findContours(
                        normalized_mask, 
                        mode,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if len(contours) > 0:
                        logger.info(f"Found {len(contours)} contours using mode {mode}")
                        hierarchy = hierarchy[0] if hierarchy is not None else []
                        contour_found = True
                        break
                except Exception as e:
                    logger.error(f"Error with contour mode {mode}: {e}")
                    continue
            
            # If still no contours, try creating a mask with clearer boundaries
            if not contour_found:
                logger.warning("No contours found, creating clearer mask")
                clear_mask = np.zeros_like(binary_mask)
                
                # Find non-zero regions
                rows, cols = np.nonzero(binary_mask)
                if len(rows) > 0 and len(cols) > 0:
                    # Calculate bounding box
                    min_row, max_row = np.min(rows), np.max(rows)
                    min_col, max_col = np.min(cols), np.max(cols)
                    
                    # Create a clearer rectangular mask
                    clear_mask[min_row:max_row+1, min_col:max_col+1] = 255
                    
                    # Try finding contours again
                    try:
                        contours, hierarchy = cv2.findContours(
                            clear_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        hierarchy = hierarchy[0] if hierarchy is not None else []
                        contour_found = True
                    except Exception as e:
                        logger.error(f"Error finding contours in clear mask: {e}")
            
            # Process contours if found
            if contour_found and 'contours' in locals() and len(contours) > 0:
                hierarchy = hierarchy if len(hierarchy) == len(contours) else [[-1, -1, -1, -1] for _ in range(len(contours))]
                
                for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                    # Filter very small contours
                    area = cv2.contourArea(contour)
                    if area < 15:
                        continue
                        
                    # Check if contour has enough points
                    if len(contour) < 3:
                        continue
                    
                    # Simplify contour
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to points list
                    points = approx.reshape(-1, 2).tolist()
                    
                    # Determine external vs hole
                    is_external = h[3] == -1
                    parent_idx = h[3] if not is_external else -1
                    
                    # Generate color
                    hue = (contour_id * 137.5) % 360
                    r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                    color = [float(r), float(g), float(b)]
                    cv_color = (int(r*255), int(g*255), int(b*255))
                    
                    # Add to results
                    all_contours.append({
                        'id': contour_id,
                        'mask_id': 0,
                        'points': points,
                        'is_external': bool(is_external),
                        'parent_idx': int(parent_idx),
                        'color': color,
                        'area': float(area)
                    })
                    
                    # For debugging
                    cv2.drawContours(debug_image, [approx], -1, cv_color, 1)
                    
                    contour_id += 1
        
        # 3. Last resort: create default contours if none were found
        if len(all_contours) == 0:
            logger.warning("Creating fallback rectangular contour")
            
            # Create a border rectangle
            margin = 20
            border_points = [
                [margin, margin],
                [width-margin, margin],
                [width-margin, height-margin],
                [margin, height-margin]
            ]
            
            all_contours.append({
                'id': 0,
                'mask_id': 0,
                'points': border_points,
                'is_external': True,
                'parent_idx': -1,
                'color': [0.8, 0.2, 0.2],  # Red
                'area': float((width-2*margin) * (height-2*margin))
            })
            
            # Draw to debug image
            cv2.rectangle(debug_image, (margin, margin), (width-margin, height-margin), (0, 0, 255), 2)
        
        # Save debug image
        debug_path = os.path.join(app.config['RESULT_FOLDER'], f"contour_debug_{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(debug_path, debug_image)
        logger.info(f"Saved contour debug image to {debug_path}")
        
        logger.info(f"Returning {len(all_contours)} contours")
        return all_contours

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

def cleanup_old_files(max_age_days=1):
    """Delete files older than max_age_days to prevent storage issues"""
    if not app.config.get('CLEANUP_FILES', False):
        return
    
    try:
        now = datetime.now(UTC)
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath), tz=UTC)
                if (now - file_modified).days >= max_age_days:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

def save_image_multiple_ways(image_data, filepath):
    """Try multiple methods to save an image to ensure it's saved correctly"""
    success = False
    errors = []
    
    # Method 1: Standard cv2.imwrite
    try:
        success = cv2.imwrite(filepath, image_data)
        if success and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Successfully saved image using cv2.imwrite")
            return True
        else:
            errors.append("cv2.imwrite returned False or file is empty")
    except Exception as e:
        errors.append(f"cv2.imwrite error: {str(e)}")
    
    # Method 2: cv2.imwrite with parameters
    if not success:
        try:
            success = cv2.imwrite(filepath, image_data, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if success and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"Successfully saved image using cv2.imwrite with parameters")
                return True
            else:
                errors.append("cv2.imwrite with parameters returned False or file is empty")
        except Exception as e:
            errors.append(f"cv2.imwrite with parameters error: {str(e)}")
    
    # Method 3: PIL
    if not success and HAS_PIL:
        try:
            pil_image = Image.fromarray(image_data)
            pil_image.save(filepath)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"Successfully saved image using PIL")
                return True
            else:
                errors.append("PIL save succeeded but file is empty")
        except Exception as e:
            errors.append(f"PIL error: {str(e)}")
    
    # Method 4: numpy.tofile for grayscale
    if not success:
        try:
            if len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1):
                # Grayscale - use PGM format
                height, width = image_data.shape[:2]
                header = f"P5\n{width} {height}\n255\n".encode()
                with open(filepath, 'wb') as f:
                    f.write(header)
                    image_data.tofile(f)
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    logger.info(f"Successfully saved image using PGM format")
                    return True
                else:
                    errors.append("PGM save succeeded but file is empty")
        except Exception as e:
            errors.append(f"PGM error: {str(e)}")
    
    # Log all errors if we failed
    if not success:
        logger.error(f"Failed to save image to {filepath}. Errors: {errors}")
        return False
    
    return success

def verify_and_regenerate_mask(job_id):
    """Verify mask exists and is valid, regenerate if needed"""
    mask_path = None
    for ext in ['png', 'pgm', 'jpg', 'bmp']:
        test_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.{ext}")
        if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
            mask_path = test_path
            break
    
    # If mask doesn't exist or is empty, try to regenerate from contours
    if mask_path is None:
        contours_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_contours.json")
        if os.path.exists(contours_path):
            try:
                with open(contours_path, 'r') as f:
                    data = json.load(f)
                
                width = data.get('image_size', {}).get('width', 800)
                height = data.get('image_size', {}).get('height', 600)
                contours_data = data.get('contours', [])
                
                # Generate mask from contours
                mask = np.zeros((height, width), dtype=np.uint8)
                for contour in contours_data:
                    points = contour.get('points', [])
                    is_external = contour.get('is_external', True)
                    
                    if len(points) >= 3:
                        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        color = 255 if is_external else 0
                        cv2.drawContours(mask, [points_array], 0, color, thickness=cv2.FILLED)
                
                # Save regenerated mask with multiple methods
                mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.png")
                if not save_image_multiple_ways(mask, mask_path):
                    # Try BMP format if PNG fails
                    mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.bmp")
                    save_image_multiple_ways(mask, mask_path)
            except Exception as e:
                logger.error(f"Error regenerating mask: {str(e)}")
    
    return mask_path

# Initialize the SAM extractor without loading model immediately
extractor = SAMExtractor(app.config['SAM_CHECKPOINT'], app.config['SAM_MODEL_TYPE'])

@app.route('/')
def index():
    """Redirect to extraction interface"""
    return redirect('/static/extraction.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.config['STATIC_FOLDER'], path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Get memory info if possible
    memory_info = {}
    try:
        if torch.cuda.is_available():
            memory_info = {
                "gpu_memory_used": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            }
        
        # System memory info
        system_mem = psutil.virtual_memory()
        memory_info.update({
            "system_memory_used": f"{system_mem.used / 1024**3:.2f} GB",
            "system_memory_total": f"{system_mem.total / 1024**3:.2f} GB",
            "system_memory_percent": f"{system_mem.percent:.1f}%"
        })
    except Exception as e:
        memory_info = {"error": str(e)}
    
    return jsonify({
        "status": "healthy", 
        "model_loaded": extractor.is_model_loaded(),
        "date": extractor.date,
        "user": extractor.user,
        "memory": memory_info,
        "config": {
            "UPLOAD_FOLDER": app.config['UPLOAD_FOLDER'],
            "RESULT_FOLDER": app.config['RESULT_FOLDER'],
            "EDITOR_CONTOURS_FOLDER": app.config['EDITOR_CONTOURS_FOLDER'],
            "UNLOAD_MODEL_AFTER_USE": app.config['UNLOAD_MODEL_AFTER_USE'],
            "MAX_IMAGE_PIXELS": app.config['MAX_IMAGE_PIXELS'],
            "USE_HALF_PRECISION": app.config['USE_HALF_PRECISION']
        }
    })

@app.route('/extract', methods=['POST'])
def extract_contours():
    """
    Extract contours from an image using SAM
    
    Parameters (form data):
    - image: The image file
    - prompt_type (optional): 'automatic' (default), 'click', or 'box'
    - points (optional): Comma-separated coordinates for click prompts (x1,y1,x2,y2,...)
    - box (optional): Comma-separated coordinates for box prompt (x1,y1,x2,y2)
    - refine (optional): Whether to refine edges using CRF (true/false)
    
    Returns:
    - JSON with contours data and metadata
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Get the image file
        file = request.files['image']
        
        # Check if file is actually an image
        if not file.filename or '.' not in file.filename:
            return jsonify({"error": "Invalid file"}), 400
            
        # Secure the filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        
        if ext not in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
            return jsonify({"error": "File must be an image (jpg, png, bmp, tif)"}), 400
        
        # Generate a unique ID for this extraction
        job_id = str(uuid.uuid4())
        
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
        file.save(image_path)
        
        # Load the image and verify it loaded correctly
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Failed to load image"}), 400
        
        # Get image dimensions
        height, width = image.shape[:2]
        logger.info(f"Processing image with dimensions {width}x{height}")
        
        # Get segmentation parameters
        prompt_type = request.form.get('prompt_type', 'automatic')
        refine_edges = request.form.get('refine', 'false').lower() == 'true'
        
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Perform segmentation based on prompt type
        if prompt_type == 'click':
            # Get point prompts
            if 'points' not in request.form:
                return jsonify({"error": "Points required for click prompt"}), 400
                
            points_str = request.form.get('points')
            try:
                coords = [float(x) for x in points_str.split(',')]
            except ValueError:
                return jsonify({"error": "Invalid point format. Use comma-separated numbers"}), 400
            
            # Ensure coordinates are in pairs
            if len(coords) % 2 != 0:
                return jsonify({"error": "Invalid point coordinates"}), 400
                
            # Reshape into pairs
            input_points = np.array(coords).reshape(-1, 2)
            
            # Get optional point labels (foreground/background)
            labels = None
            if 'labels' in request.form:
                labels_str = request.form.get('labels')
                try:
                    labels = np.array([int(x) for x in labels_str.split(',')])
                except ValueError:
                    return jsonify({"error": "Invalid labels format. Use comma-separated 0s and 1s"}), 400
                
            mask = extractor.segment_with_points(image, input_points, labels)
            
        elif prompt_type == 'box':
            # Get box prompt
            if 'box' not in request.form:
                return jsonify({"error": "Box coordinates required for box prompt"}), 400
                
            box_str = request.form.get('box')
            try:
                coords = [float(x) for x in box_str.split(',')]
            except ValueError:
                return jsonify({"error": "Invalid box format. Use comma-separated numbers"}), 400
            
            # Ensure we have 4 coordinates
            if len(coords) != 4:
                return jsonify({"error": "Box must have 4 coordinates (x1,y1,x2,y2)"}), 400
                
            # Reshape into corner points
            box = np.array([[coords[0], coords[1]], [coords[2], coords[3]]])
            
            mask = extractor.segment_with_box(image, box)
            
        else:  # automatic
            mask = extractor.segment_automatic(image)
        
        # Apply edge refinement if requested
        if refine_edges:
            mask = extractor.refine_edges(image, mask)
        
        # Ensure mask has expected dimensions
        if mask.shape[:2] != (height, width):
            logger.warning(f"Resizing mask from {mask.shape} to {(height, width)}")
            mask = cv2.resize(mask.astype(np.uint8), (width, height)) > 0
        
        # Make sure mask is not empty
        if np.sum(mask) == 0:
            logger.warning("Empty mask detected, creating basic mask")
            # Create a basic mask in the center of the image
            center_mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
            mask = center_mask > 0

        # Convert mask to uint8 for saving (0 or 255)
        mask_image = (mask.astype(np.uint8) * 255)
        
        # Save mask to disk using multiple methods to ensure it works
        mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.png")
        save_success = save_image_multiple_ways(mask_image, mask_path)
        
        if not save_success:
            logger.error("Failed to save mask image, trying BMP format")
            mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.bmp")
            save_success = save_image_multiple_ways(mask_image, mask_path)
        
        # Verify mask was saved
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            logger.error("Could not save mask image - saving debug version")
            debug_mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask_debug.npy")
            np.save(debug_mask_path, mask_image)
            mask_path = debug_mask_path
        
        logger.info(f"Mask saved to {mask_path}")
        
        # Create a visualization of the mask
        overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay.png")
        
        try:
            # Create a colored overlay
            overlay = image.copy()
            
            # Apply color using numpy where to avoid loops
            mask_bool = mask > 0
            for c in range(3):
                channel = overlay[:,:,c].copy()
                if c == 1:  # Green channel - make more prominent
                    overlay[:,:,c] = np.where(mask_bool, (channel * 0.3 + 179).astype(np.uint8), channel)
                else:  # Red and Blue channels - darken slightly
                    overlay[:,:,c] = np.where(mask_bool, (channel * 0.7).astype(np.uint8), channel)
            
            # Add contour outlines for better visibility
            contours_cv, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours_cv, -1, (0, 255, 255), 2)  # Yellow outlines
            
            # Save overlay image
            save_success = save_image_multiple_ways(overlay, overlay_path)
            
            if not save_success:
                logger.warning("Failed to save overlay image, trying alternate format")
                overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay.jpg")
                save_image_multiple_ways(overlay, overlay_path)
                
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Create a very simple overlay if the above fails
            try:
                simple_overlay = image.copy()
                mask_rgb = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                simple_overlay = cv2.addWeighted(simple_overlay, 0.7, mask_rgb, 0.3, 0)
                overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay_simple.png")
                save_image_multiple_ways(simple_overlay, overlay_path)
            except Exception as e2:
                logger.error(f"Error creating simple overlay: {str(e2)}")
        
        # Extract contours
        contours = extractor.extract_contours(mask)
        logger.info(f"Extracted {len(contours)} contours")
        
        # Return mask as base64 for preview
        try:
            # Try to read back the saved mask
            saved_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if saved_mask is None or saved_mask.max() == 0:
                # If failed, use the original mask
                saved_mask = mask_image
                
            _, buffer = cv2.imencode('.png', saved_mask)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding mask as base64: {str(e)}")
            # Create a simple placeholder if encoding fails
            placeholder = np.zeros((100, 100), dtype=np.uint8)
            placeholder[25:75, 25:75] = 255  # White square
            _, buffer = cv2.imencode('.png', placeholder)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create result data with updated date/time and user
        result = {
            "job_id": job_id,
            "image_size": {"width": int(width), "height": int(height)},
            "contours": contours,
            "mask_base64": mask_b64,
            "metadata": {
                "date": "2025-05-27 10:12:00",  # Updated timestamp
                "user": "FETHl",  # Updated user
                "prompt_type": prompt_type,
                "refinement": bool(refine_edges),
                "num_contours": len(contours),
                "num_masks": len(extractor.sam_masks) if hasattr(extractor, 'sam_masks') and extractor.sam_masks else 0
            }
        }
        
        # Save contours to a JSON file for the second API to use
        contours_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_contours.json")
        with open(contours_path, 'w') as f:
            json.dump(result, f, cls=NumpyJSONEncoder)
        
        # Also save to the editor contours folder for direct access
        editor_contours_path = os.path.join(app.config['EDITOR_CONTOURS_FOLDER'], f"{job_id}_contours.json")
        with open(editor_contours_path, 'w') as f:
            json.dump(result, f, cls=NumpyJSONEncoder)
        
        # Copy the image to the editor uploads folder
        try:
            editor_image_path = os.path.join(app.config['EDITOR_UPLOADS_FOLDER'], f"{job_id}.{ext}")
            shutil.copy(image_path, editor_image_path)
        except Exception as e:
            logger.error(f"Error copying image to editor folder: {str(e)}")
        
        # Clean up old files occasionally
        if np.random.random() < 0.1:  # 10% chance to trigger cleanup
            cleanup_old_files()
            
        # Clear stored masks to free memory
        if hasattr(extractor, 'sam_masks'):
            extractor.sam_masks = None
            
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        logger.info(f"Extraction completed: {len(contours)} contours extracted")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clear stored masks in case of error
        if hasattr(extractor, 'sam_masks'):
            extractor.sam_masks = None
            
        # Make sure model is unloaded in case of error
        if app.config['UNLOAD_MODEL_AFTER_USE'] and extractor.is_model_loaded():
            extractor.unload_model()
            
        # Clear GPU cache after error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return jsonify({"error": str(e)}), 500
    


@app.route('/mask/<job_id>', methods=['GET'])
def get_mask(job_id):
    """Get the binary mask for a job"""
    # Validate job_id to prevent path traversal
    if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
        return jsonify({"error": "Invalid job ID"}), 400
    
    # First try to find existing mask
    mask_path = verify_and_regenerate_mask(job_id)
    
    if mask_path and os.path.exists(mask_path):
        return send_file(mask_path)
    
    return jsonify({"error": "Mask not found and could not be regenerated"}), 404

@app.route('/overlay/<job_id>', methods=['GET'])
def get_overlay(job_id):
    """Get the visualization overlay for a job"""
    # Validate job_id to prevent path traversal
    if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
        return jsonify({"error": "Invalid job ID"}), 400
        
    # Check for existing overlay
    for ext in ['png', 'jpg', 'bmp']:
        overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay.{ext}")
        if os.path.exists(overlay_path) and os.path.getsize(overlay_path) > 0:
            return send_file(overlay_path)
    
    # Try to regenerate overlay
    try:
        # Find the image
        image_path = None
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        # Get or regenerate the mask
        mask_path = verify_and_regenerate_mask(job_id)
        
        if not image_path or not mask_path:
            return jsonify({"error": "Missing image or mask for overlay generation"}), 404
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return jsonify({"error": "Failed to load image or mask"}), 500
        
        # Ensure mask dimensions match image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create colored overlay
        overlay = image.copy()
        mask_bool = mask > 127
        
        # Apply color using numpy where to avoid loops
        for c in range(3):
            channel = overlay[:,:,c].copy()
            if c == 1:  # Green channel
                overlay[:,:,c] = np.where(mask_bool, (channel * 0.3 + 179).astype(np.uint8), channel)
            else:  # Red and Blue channels
                overlay[:,:,c] = np.where(mask_bool, (channel * 0.7).astype(np.uint8), channel)
        
        # Add contour outlines for better visibility
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Yellow outlines
        
        # Save regenerated overlay
        overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay.png")
        save_image_multiple_ways(overlay, overlay_path)
        
        if os.path.exists(overlay_path) and os.path.getsize(overlay_path) > 0:
            logger.info(f"Successfully regenerated overlay for {job_id}")
            return send_file(overlay_path)
        else:
            return jsonify({"error": "Failed to save regenerated overlay"}), 500
            
    except Exception as e:
        logger.error(f"Error regenerating overlay: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error generating overlay: {str(e)}"}), 500

@app.route('/contours/<job_id>', methods=['GET'])
def get_contours(job_id):
    """Get the extracted contours for a job"""
    # Validate job_id to prevent path traversal
    if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
        return jsonify({"error": "Invalid job ID"}), 400
        
    contours_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_contours.json")
    
    if not os.path.exists(contours_path):
        return jsonify({"error": "Contours not found"}), 404
        
    with open(contours_path, 'r') as f:
        contours_data = json.load(f)
    
    # Update metadata with current date and user
    if 'metadata' not in contours_data:
        contours_data['metadata'] = {}
    
    contours_data['metadata']['date'] = "2025-05-27 10:12:00"
    contours_data['metadata']['user'] = "FETHl"
        
    return jsonify(contours_data)

@app.route('/copy_contours/<job_id>', methods=['POST'])
def copy_contours_to_editor(job_id):
    """
    Copy contours from SAM API to Editor API
    """
    try:
        # Validate job_id
        if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
            return jsonify({"error": "Invalid job ID"}), 400
            
        # Check if contours exist
        contours_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_contours.json")
        if not os.path.exists(contours_path):
            return jsonify({"error": "Contours not found"}), 404
            
        # Load contours
        with open(contours_path, 'r') as f:
            contours_data = json.load(f)
        
        # Update metadata
        if 'metadata' not in contours_data:
            contours_data['metadata'] = {}
            
        contours_data['metadata']['date'] = "2025-05-27 10:12:00"
        contours_data['metadata']['user'] = "FETHl"
        
        # Copy to editor contours directory
        editor_contours_dir = app.config['EDITOR_CONTOURS_FOLDER']
        os.makedirs(editor_contours_dir, exist_ok=True)
        
        editor_path = os.path.join(editor_contours_dir, f"{job_id}_contours.json")
        with open(editor_path, 'w') as f:
            json.dump(contours_data, f, cls=NumpyJSONEncoder)
        
        # Also copy the image if it exists
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
            if os.path.exists(image_path):
                # Copy to the uploads folder of the editor API
                editor_upload_dir = app.config['EDITOR_UPLOADS_FOLDER']
                os.makedirs(editor_upload_dir, exist_ok=True)
                
                editor_image_path = os.path.join(editor_upload_dir, f"{job_id}.{ext}")
                # Use copy instead of copy2 to avoid the "same file" error
                shutil.copy(image_path, editor_image_path)
                
                logger.info(f"Copied image {image_path} to {editor_image_path}")
                break
        
        # Also copy the mask for reference
        mask_path = verify_and_regenerate_mask(job_id)
        if mask_path:
            # Copy to the editor's result folder
            mask_ext = os.path.splitext(mask_path)[1]
            editor_mask_path = os.path.join(editor_contours_dir, f"{job_id}_mask{mask_ext}")
            shutil.copy(mask_path, editor_mask_path)
            logger.info(f"Copied mask {mask_path} to {editor_mask_path}")
        
        logger.info(f"Contours and related files copied to editor API for job {job_id}")
        
        return jsonify({
            "status": "success",
            "message": "Contours copied to editor successfully",
            "path": editor_path
        })
        
    except Exception as e:
        logger.error(f"Error copying contours: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/image/<job_id>', methods=['GET'])
def get_image(job_id):
    """Get the original image for a job"""
    # Validate job_id to prevent path traversal
    if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
        return jsonify({"error": "Invalid job ID"}), 400
    
    # Try different image extensions
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
        if os.path.exists(image_path):
            return send_file(image_path)
    
    return jsonify({"error": "Image not found"}), 404

@app.route('/regenerate/<job_id>', methods=['POST'])
def regenerate_mask_from_contours_endpoint(job_id):
    """
    Regenerate a mask from contours if mask file is missing or corrupt
    """
    # Validate job_id
    if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
        return jsonify({"error": "Invalid job ID"}), 400
        
    contours_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_contours.json")
    
    if not os.path.exists(contours_path):
        return jsonify({"error": "Contours not found"}), 404
    
    try:
        # Load contours
        with open(contours_path, 'r') as f:
            data = json.load(f)
        
        width = data.get('image_size', {}).get('width', 800)
        height = data.get('image_size', {}).get('height', 600)
        contours_data = data.get('contours', [])
        
        # Generate mask from contours
        mask = np.zeros((height, width), dtype=np.uint8)
        for contour in contours_data:
            points = contour.get('points', [])
            is_external = contour.get('is_external', True)
            
            if len(points) >= 3:
                points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                color = 255 if is_external else 0
                cv2.drawContours(mask, [points_array], 0, color, thickness=cv2.FILLED)
        
        # Save regenerated mask
        mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_mask.png")
        save_success = save_image_multiple_ways(mask, mask_path)
        
        if not save_success:
            return jsonify({"error": "Failed to save regenerated mask"}), 500
        
        # Update metadata in contours file
        data['metadata']['date'] = "2025-05-27 10:12:00"
        data['metadata']['user'] = "FETHl"
        data['metadata']['regenerated'] = True
        
        with open(contours_path, 'w') as f:
            json.dump(data, f, cls=NumpyJSONEncoder)
        
        # Also update editor's contours file if it exists
        editor_contours_path = os.path.join(app.config['EDITOR_CONTOURS_FOLDER'], f"{job_id}_contours.json")
        if os.path.exists(editor_contours_path):
            with open(editor_contours_path, 'r') as f:
                editor_data = json.load(f)
            
            editor_data['metadata']['date'] = "2025-05-27 10:12:00"
            editor_data['metadata']['user'] = "FETHl"
            editor_data['metadata']['regenerated'] = True
            
            with open(editor_contours_path, 'w') as f:
                json.dump(editor_data, f, cls=NumpyJSONEncoder)
        
        # Try to regenerate overlay as well
        try:
            regenerate_overlay(job_id)
        except Exception as e:
            logger.warning(f"Error regenerating overlay: {e}")
        
        return jsonify({
            "status": "success",
            "message": "Mask regenerated from contours",
            "path": mask_path
        })
        
    except Exception as e:
        logger.error(f"Error regenerating mask: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def regenerate_overlay(job_id):
    """Helper function to regenerate overlay from image and mask"""
    # Find image
    image_path = None
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
        if os.path.exists(test_path):
            image_path = test_path
            break
    
    # Find mask
    mask_path = verify_and_regenerate_mask(job_id)
    
    if not image_path or not mask_path:
        logger.error(f"Cannot regenerate overlay: missing image or mask for job {job_id}")
        return False
    
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        logger.error(f"Failed to load image or mask for job {job_id}")
        return False
    
    # Ensure mask dimensions match image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored overlay
    overlay = image.copy()
    mask_bool = mask > 127
    
    # Apply color using numpy where to avoid loops
    for c in range(3):
        channel = overlay[:,:,c].copy()
        if c == 1:  # Green channel
            overlay[:,:,c] = np.where(mask_bool, (channel * 0.3 + 179).astype(np.uint8), channel)
        else:  # Red and Blue channels
            overlay[:,:,c] = np.where(mask_bool, (channel * 0.7).astype(np.uint8), channel)
    
    # Add contour outlines for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Yellow outlines
    
    # Save regenerated overlay
    overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{job_id}_overlay.png")
    save_success = save_image_multiple_ways(overlay, overlay_path)
    
    return save_success

if __name__ == '__main__':
        # Update date and user
    extractor.date = "2025-05-27 10:12:00"  # Updated with latest timestamp
    extractor.user = "FETHl"
    
    logger.info(f"Starting SAM Extraction API (memory-optimized) with date: {extractor.date}, user: {extractor.user}")
    logger.info(f"Model will be loaded on demand: {app.config['UNLOAD_MODEL_AFTER_USE']}")
    logger.info("Using FP32 precision for all operations (FP16/mixed precision disabled)")
    
    # Use threaded server for better performance
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5001)), 
        debug=app.config['DEBUG'], 
        threaded=True
    )