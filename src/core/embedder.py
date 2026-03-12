import logging
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

_app_instance = None


def _get_face_app():
    global _app_instance
    if _app_instance is None:
        from insightface.app import FaceAnalysis

        log.info("Loading InsightFace buffalo_l model (first run downloads ~300 MB)...")
        _app_instance = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app_instance.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("Model ready.")
    return _app_instance


def get_embedding(image_path: str) -> np.ndarray | None:
    """Extract the 512-d normalized face embedding from the largest face in the image.

    Returns None if no face is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return get_embedding_from_array(img)


def get_embedding_from_array(img: np.ndarray) -> np.ndarray | None:
    """Extract embedding from a BGR numpy array (OpenCV format)."""
    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = largest.embedding
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def get_all_embeddings(image_path: str) -> list[dict]:
    """Return embeddings for every detected face with bounding box info."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    app = _get_face_app()
    faces = app.get(img)
    results = []
    for face in faces:
        emb = face.embedding.copy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        results.append({
            "embedding": emb.astype(np.float32),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
        })
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def batch_embed_folder(
    folder: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Embed all images in a folder. Returns list of {path, embedding}.

    progress_callback(current, total, current_path) is called per image.
    cancel_check() should return True to abort.
    """
    import os
    from utils.image_utils import SUPPORTED_EXTENSIONS

    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, total, path)
        try:
            emb = get_embedding(path)
            if emb is not None:
                results.append({"path": path, "embedding": emb})
        except Exception as e:
            log.warning("Failed to embed %s: %s", path, e)
    return results
