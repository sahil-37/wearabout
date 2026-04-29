"""
Recommendation endpoints
"""

import logging
import os
import tempfile
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.engines import RecommendationEngine
from app.utils.validators import validate_image_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize recommendation engine
recommendation_engine = None


def get_engine():
    """Get or create recommendation engine instance"""
    global recommendation_engine
    if recommendation_engine is None:
        recommendation_engine = RecommendationEngine()
    return recommendation_engine


@router.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    gender: str = Query("unisex", enum=settings.GENDER_OPTIONS),
    top_k: Optional[int] = Query(None, ge=1, le=100)
) -> Dict:
    """
    Get fashion recommendations for an image

    Args:
        file: Image file of person wearing outfit
        gender: Gender filter (men, women, unisex, n.a.)
        top_k: Number of recommendations

    Returns:
        Recommendations
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
            # Get recommendations
            engine = get_engine()
            result = engine.recommend(
                img_path=tmp_path,
                gender=gender,
                top_k=top_k
            )

            return result

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-similar")
async def find_similar(
    file: UploadFile = File(...),
    top_k: Optional[int] = Query(None, ge=1, le=100)
) -> Dict:
    """
    Find similar fashion items (without full-body check)

    Args:
        file: Image file to find similar items
        top_k: Number of similar items

    Returns:
        Similar items
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
            # Find similar items
            engine = get_engine()
            result = engine.find_similar_images(
                img_path=tmp_path,
                top_k=top_k
            )

            return result

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendation-categories")
async def get_categories() -> Dict:
    """Get available fashion categories"""
    return {
        "categories": settings.FASHION_CATEGORIES,
        "gender_options": settings.GENDER_OPTIONS
    }


@router.get("/recommendation-config")
async def get_config() -> Dict:
    """Get recommendation engine configuration"""
    return {
        "max_recommendations": settings.MAX_RECOMMENDATIONS,
        "min_recommendations": settings.MIN_RECOMMENDATIONS,
        "default_k_neighbors": settings.DEFAULT_K_NEIGHBORS,
        "categories": settings.FASHION_CATEGORIES,
        "gender_options": settings.GENDER_OPTIONS
    }
