"""
Fashion Recommendation API - Main Application
Buy Me That Look API Server
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # prevent OpenMP conflict on macOS

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from app.routes import auth, detection, features, health, recommendation, validation
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

CATALOG_ROOT = Path("/Users/sahil/Desktop/AI Ebooks/Buy Me That Look/Product")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Fashion Recommendation API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"DEBUG: {settings.DEBUG}")
    yield
    logger.info("Shutting down Fashion Recommendation API")


app = FastAPI(
    title="Buy Me That Look - Fashion Recommendation API",
    description="AI-powered fashion recommendation system using computer vision",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,          prefix="/api/v1",      tags=["Health"])
app.include_router(auth.router,            prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(validation.router,      prefix="/api/v1",      tags=["Validation"])
app.include_router(detection.router,       prefix="/api/v1",      tags=["Detection"])
app.include_router(features.router,        prefix="/api/v1",      tags=["Features"])
app.include_router(recommendation.router,  prefix="/api/v1",      tags=["Recommendation"])

app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], tags=["Monitoring"])


@app.get("/product-image", tags=["Assets"])
async def product_image(path: str = Query(...)):
    """Serve a local catalog product image by absolute path."""
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if not str(p).startswith(str(CATALOG_ROOT)):
        raise HTTPException(status_code=403, detail="Forbidden")
    return FileResponse(p)


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "message": "Fashion Recommendation API - Buy Me That Look",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
        "metrics": "/metrics",
    }


# Serve built frontend (production — `make frontend-build` first)
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
