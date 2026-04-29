"""
Recommendation Engine — FAISS-backed similarity search with ResNet-50 features.
"""

import json
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models import FeatureExtractor, ObjectDetector, PoseEstimation

logger = logging.getLogger(__name__)

CATEGORIES = ["topwear", "bottomwear", "footwear", "eyewear", "handbag"]

# Maps YOLO class names → catalog category keys
YOLO_TO_CATEGORY = {
    "Topwear":   "topwear",
    "Bottomwear": "bottomwear",
    "Footwear":  "footwear",
    "Eyewear":   "eyewear",
    "Handbag":   "handbag",
}


class RecommendationEngine:

    def __init__(self):
        self.pose_estimator   = PoseEstimation()
        self.feature_extractor = FeatureExtractor()
        self.object_detector  = None

        # catalog
        self.metadata: List[Dict]          = []
        self.features_array: Optional[np.ndarray] = None
        self.image_paths: List[str]        = []
        self.feature_categories: List[str] = []

        # FAISS indices — one global + one per category
        self.faiss_index          = None
        self.faiss_by_category: Dict = {}   # category → (faiss_index, list[original_idx])

        self._loaded = False

        self._initialize_detector()

    # ── initialization ────────────────────────────────────────────────────────

    def _initialize_detector(self):
        try:
            self.object_detector = ObjectDetector(
                weights_path=settings.YOLO_WEIGHTS_PATH,
                conf_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
                iou_threshold=settings.NMS_IOU_THRESHOLD,
            )
        except Exception as e:
            logger.warning(f"Detector init failed: {e}")

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            self._load_features()
            self._load_metadata()
            self._build_faiss_indices()
            self._loaded = True
        except Exception as e:
            logger.error(f"Catalog load failed: {e}")

    def _load_features(self):
        with open(settings.FEATURES_PATH, "rb") as f:
            data = pickle.load(f)

        self.features_array      = data["features"].astype(np.float32)
        self.image_paths         = data["image_paths"]
        self.feature_categories  = data["categories"]
        logger.info(f"Loaded {len(self.image_paths)} feature vectors  shape={self.features_array.shape}")

    def _load_metadata(self):
        with open(settings.METADATA_PATH) as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} metadata records")

    def _build_faiss_indices(self):
        import faiss

        _, d = self.features_array.shape

        # global index (inner-product on L2-normalised vectors == cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.features_array)
        logger.info(f"Global FAISS index: {self.faiss_index.ntotal} vectors")

        # per-category indices
        for cat in CATEGORIES:
            idxs = [i for i, c in enumerate(self.feature_categories) if c == cat]
            if not idxs:
                continue
            cat_feats = self.features_array[idxs]
            idx_obj = faiss.IndexFlatIP(d)
            idx_obj.add(cat_feats)
            self.faiss_by_category[cat] = (idx_obj, idxs)
            logger.info(f"  {cat}: {idx_obj.ntotal} vectors")

    # ── public endpoints ──────────────────────────────────────────────────────

    def recommend(
        self,
        img_path: str,
        gender: str = "unisex",
        top_k: Optional[int] = None,
    ) -> Dict:
        self._ensure_loaded()
        top_k = top_k or settings.MAX_RECOMMENDATIONS

        # 1 — detect clothing items first (primary gate)
        detections = []
        using_mock = False
        if self.object_detector:
            det_result = self.object_detector.detect(img_path)
            if det_result.get("success"):
                detections = det_result["detections"]
                # check if the fallback mock detector is active
                using_mock = self.object_detector.model and \
                    isinstance(self.object_detector.model, dict) and \
                    self.object_detector.model.get("type") == "mock"

        real_detections = [d for d in detections if not d.get("note") == "mock"]

        # 2 — pose check (used only as fallback signal when no clothing detected)
        is_full_shot, *_ = self.pose_estimator.pose_estimation(
            img_path, confidence_threshold=settings.POSE_CONFIDENCE_THRESHOLD
        )

        # 3 — reject if no clothing found AND no person detected
        # This blocks thumbprints, food, random objects etc.
        # Allow through if: real clothing detected OR person found by HOG
        has_clothing = len(real_detections) > 0
        if not has_clothing and not is_full_shot and not using_mock:
            return {
                "success": False,
                "error":   "No clothing or person detected. Please upload a photo of an outfit or clothing item."
            }

        # 4 — extract query features
        query_vec = self.feature_extractor.extract_features(img_path)
        if query_vec is None:
            return {"success": False, "error": "Feature extraction failed"}

        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # 5 — search per detected category, fall back to global search
        detected_categories = list({
            YOLO_TO_CATEGORY[d["class"]]
            for d in real_detections
            if d["class"] in YOLO_TO_CATEGORY
        })

        recommendations = []
        seen_ids = set()

        if detected_categories:
            per_cat_k = max(top_k // len(detected_categories), settings.MIN_RECOMMENDATIONS)
            for cat in detected_categories:
                hits = self._search_category(query_vec, cat, per_cat_k, gender, seen_ids)
                recommendations.extend(hits)
                seen_ids.update(h["id"] for h in hits)
        else:
            recommendations = self._search_global(query_vec, top_k, gender, seen_ids)

        return {
            "success":            True,
            "full_shot_detected": is_full_shot,
            "detections":         real_detections,
            "recommendations":    recommendations[:top_k],
        }

    def find_similar_images(
        self,
        img_path: str,
        top_k: Optional[int] = None,
    ) -> Dict:
        self._ensure_loaded()
        top_k = top_k or settings.MAX_RECOMMENDATIONS

        query_vec = self.feature_extractor.extract_features(img_path)
        if query_vec is None:
            return {"success": False, "error": "Feature extraction failed"}

        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        items = self._search_global(query_vec, top_k, gender="unisex", seen_ids=set())

        return {"success": True, "similar_items": items}

    # ── FAISS search helpers ──────────────────────────────────────────────────

    def _search_global(
        self,
        query: np.ndarray,
        k: int,
        gender: str,
        seen_ids: set,
    ) -> List[Dict]:
        if self.faiss_index is None:
            return []

        k_fetch = min(k * 4, self.faiss_index.ntotal)   # over-fetch to allow filtering
        scores, idxs = self.faiss_index.search(query, k_fetch)

        return self._build_results(idxs[0], scores[0], k, gender, seen_ids)

    def _search_category(
        self,
        query: np.ndarray,
        category: str,
        k: int,
        gender: str,
        seen_ids: set,
    ) -> List[Dict]:
        if category not in self.faiss_by_category:
            return self._search_global(query, k, gender, seen_ids)

        cat_index, orig_idxs = self.faiss_by_category[category]
        k_fetch = min(k * 4, cat_index.ntotal)
        scores, local_idxs = cat_index.search(query, k_fetch)

        # map local indices back to global catalog indices
        global_idxs = [orig_idxs[i] for i in local_idxs[0] if i < len(orig_idxs)]
        return self._build_results(global_idxs, scores[0], k, gender, seen_ids)

    def _build_results(
        self,
        idxs,
        scores,
        k: int,
        gender: str,
        seen_ids: set,
    ) -> List[Dict]:
        results = []
        for idx, score in zip(idxs, scores):
            if len(results) >= k:
                break
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]
            item_id = str(idx)
            if item_id in seen_ids:
                continue

            item_gender = item.get("gender", "n.a.").lower()
            if gender != "unisex" and item_gender not in (gender.lower(), "unisex", "n.a."):
                continue

            results.append({
                "id":               item_id,
                "name":             item.get("name", ""),
                "brand":            item.get("brand", ""),
                "category":         item.get("category", ""),
                "price":            item.get("price", 0),
                "gender":           item_gender,
                "similarity_score": round(float(score), 4),
                "product_url":      item.get("product_url", ""),
                "image_path":       item.get("image_path", ""),
            })

        return results

    def cleanup(self):
        if self.pose_estimator:
            self.pose_estimator.cleanup()
        if self.object_detector:
            self.object_detector.cleanup()
        if self.feature_extractor:
            self.feature_extractor.cleanup()
