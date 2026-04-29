"""
Feature extraction endpoints
"""

import logging
import os
import tempfile
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models import FeatureExtractor
from app.utils.validators import validate_image_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize extractor
feature_extractor = None


def get_extractor():
    """Get or create feature extractor instance"""
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = FeatureExtractor(
            input_size=(settings.RESNET_INPUT_SIZE, settings.RESNET_INPUT_SIZE)
        )
    return feature_extractor


@router.post("/extract-features")
async def extract_features(file: UploadFile = File(...)) -> Dict:
    """
    Extract features from image using ResNet-50

    Args:
        file: Image file to process

    Returns:
        Feature vector
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
            # Extract features
            extractor = get_extractor()

            features = extractor.extract_features(tmp_path)

            if features is None:
                return {
                    "success": False,
                    "error": "Failed to extract features"
                }

            return {
                "success": True,
                "feature_dimension": int(len(features)),
                "features": [float(f) for f in features]
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-batch-features")
async def extract_batch_features(
    files: List[UploadFile] = File(...)
) -> Dict:
    """
    Extract features from multiple images

    Args:
        files: Image files to process

    Returns:
        Feature vectors
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files allowed")

        temp_paths = []
        results = []

        try:
            # Save temporary files
            for file in files:
                error = validate_image_file(file.filename, file.size)
                if error:
                    raise HTTPException(status_code=400, detail=error)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    contents = await file.read()
                    tmp.write(contents)
                    temp_paths.append(tmp.name)

            # Extract features
            extractor = get_extractor()

            features_list = extractor.extract_batch_features(temp_paths)

            for i, features in enumerate(features_list):
                results.append({
                    "file_index": int(i),
                    "filename": files[i].filename,
                    "success": bool(features is not None),
                    "features": [float(f) for f in features] if features is not None else None
                })

            return {
                "success": True,
                "total_files": len(files),
                "processed_files": len([r for r in results if r["success"]]),
                "results": results
            }

        finally:
            for tmp_path in temp_paths:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch feature extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
