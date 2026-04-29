"""
Image validation endpoints
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict
import logging
import os
import tempfile

from app.models import PoseEstimation
from app.config import settings
from app.utils.validators import validate_image_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize pose estimator
pose_estimator = None


def get_pose_estimator():
    """Get or create pose estimator instance"""
    global pose_estimator
    if pose_estimator is None:
        pose_estimator = PoseEstimation()
    return pose_estimator


@router.post("/validate-image")
async def validate_image(file: UploadFile = File(...)) -> Dict:
    """
    Validate if image is a full-body shot

    Args:
        file: Image file to validate

    Returns:
        Validation result
    """
    try:
        # Validate file
        error = validate_image_file(file.filename, file.size)
        if error:
            raise HTTPException(status_code=400, detail=error)

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Run pose estimation
            pose_est = get_pose_estimator()
            is_full_shot, _, _ = pose_est.pose_estimation(
                tmp_path,
                confidence_threshold=settings.POSE_CONFIDENCE_THRESHOLD
            )

            return {
                "valid": is_full_shot,
                "full_body_detected": is_full_shot,
                "message": "Full-body shot detected" if is_full_shot else "Not a full-body shot"
            }

        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-image-quality")
async def check_image_quality(file: UploadFile = File(...)) -> Dict:
    """
    Check image quality and properties

    Args:
        file: Image file to check

    Returns:
        Image quality metrics
    """
    try:
        import cv2

        # Validate file
        error = validate_image_file(file.filename, file.size)
        if error:
            raise HTTPException(status_code=400, detail=error)

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Read image
            img = cv2.imread(tmp_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            height, width, channels = img.shape

            # Calculate quality metrics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            return {
                "width": width,
                "height": height,
                "channels": channels,
                "aspect_ratio": round(width / height, 2),
                "blur_detection": float(laplacian_var),
                "is_blurry": laplacian_var < 100,
                "file_size_mb": file.size / (1024 * 1024)
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking image quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
