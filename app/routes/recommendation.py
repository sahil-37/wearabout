"""
Recommendation endpoints
"""

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.engines import RecommendationEngine
from app.utils.validators import validate_image_file

logger = logging.getLogger(__name__)
router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=4)

recommendation_engine = None


def get_engine() -> RecommendationEngine:
    global recommendation_engine
    if recommendation_engine is None:
        recommendation_engine = RecommendationEngine()
    return recommendation_engine


async def _run(fn, *args):
    """Run a blocking function in the threadpool so the event loop isn't blocked."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


@router.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    gender: str = Query("unisex", enum=settings.GENDER_OPTIONS),
    top_k: Optional[int] = Query(None, ge=1, le=100),
) -> Dict:
    """Get fashion recommendations for an uploaded outfit photo."""
    error = validate_image_file(file.filename, file.size)
    if error:
        raise HTTPException(status_code=400, detail=error)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        engine = get_engine()
        result = await _run(engine.recommend, tmp_path, gender, top_k)
        return result
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/find-similar")
async def find_similar(
    file: UploadFile = File(...),
    top_k: Optional[int] = Query(None, ge=1, le=100),
) -> Dict:
    """Find similar fashion items without a full-body pose check."""
    error = validate_image_file(file.filename, file.size)
    if error:
        raise HTTPException(status_code=400, detail=error)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        engine = get_engine()
        result = await _run(engine.find_similar_images, tmp_path, top_k)
        return result
    except Exception as e:
        logger.error(f"Find similar error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/recommendation-categories")
async def get_categories() -> Dict:
    return {
        "categories": settings.FASHION_CATEGORIES,
        "gender_options": settings.GENDER_OPTIONS,
    }


@router.get("/recommendation-config")
async def get_config() -> Dict:
    return {
        "max_recommendations": settings.MAX_RECOMMENDATIONS,
        "min_recommendations": settings.MIN_RECOMMENDATIONS,
        "default_k_neighbors": settings.DEFAULT_K_NEIGHBORS,
        "categories": settings.FASHION_CATEGORIES,
        "gender_options": settings.GENDER_OPTIONS,
    }
