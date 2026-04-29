"""
Object detection endpoints
"""

import logging
import os
import tempfile
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models import ObjectDetector
from app.utils.validators import validate_image_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize detector
object_detector = None


def get_detector():
    """Get or create detector instance"""
    global object_detector
    if object_detector is None:
        object_detector = ObjectDetector(
            weights_path=settings.YOLO_WEIGHTS_PATH,
            conf_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=settings.NMS_IOU_THRESHOLD
        )
    return object_detector


@router.post("/detect-items")
async def detect_items(file: UploadFile = File(...)) -> Dict:
    """
    Detect fashion items in image

    Args:
        file: Image file to process

    Returns:
        Detection results
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
            # Run detection
            detector = get_detector()
            if detector.model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Object detection model not available"
                )

            result = detector.detect(tmp_path)

            return {
                "success": result.get("success", False),
                "detections": result.get("detections", []),
                "num_detections": len(result.get("detections", []))
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-with-visualization")
async def detect_with_visualization(file: UploadFile = File(...)) -> Dict:
    """
    Detect items and return annotated image

    Args:
        file: Image file to process

    Returns:
        Detection results with base64 encoded image
    """
    try:
        import cv2
        import base64

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
            # Run detection
            detector = get_detector()
            if detector.model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Object detection model not available"
                )

            result = detector.detect(tmp_path)

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error"))

            # Draw detections
            annotated_img = detector.draw_detections(
                tmp_path,
                result["detections"]
            )

            if annotated_img is None:
                raise HTTPException(status_code=500, detail="Failed to annotate image")

            # Encode image
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode()

            return {
                "success": True,
                "detections": result.get("detections", []),
                "num_detections": len(result.get("detections", [])),
                "annotated_image": f"data:image/jpeg;base64,{img_base64}"
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detection with visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
