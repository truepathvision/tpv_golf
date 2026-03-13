import logging
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

_app_instance = None
_active_model: str = "buffalo_l"

AVAILABLE_MODELS = {
    "buffalo_l": "ResNet50 @ WebFace600K — 98.23% AgeDB-30 (default, ~300 MB)",
    "antelopev2": "ResNet100 @ Glint360K — higher accuracy, better cross-age (~400 MB)",
    "buffalo_s": "MobileFaceNet @ WebFace600K — lightweight (~160 MB)",
}

DET_SCORE_THRESHOLD = 0.5
TTA_BRIGHTNESS_DELTAS = [-20, 20]


def get_active_model() -> str:
    return _active_model


def set_model(name: str):
    """Switch the recognition model. Resets the cached instance so it reloads on next use."""
    global _active_model, _app_instance
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(AVAILABLE_MODELS)}")
    if name != _active_model or _app_instance is None:
        _active_model = name
        _app_instance = None
        log.info("Model set to %s (will load on next use)", name)


def _get_face_app():
    global _app_instance
    if _app_instance is None:
        from insightface.app import FaceAnalysis

        log.info("Loading InsightFace %s model...", _active_model)
        _app_instance = FaceAnalysis(name=_active_model, providers=["CPUExecutionProvider"])
        _app_instance.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("Model %s ready.", _active_model)
    return _app_instance


def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _select_best_face(faces, min_score: float = DET_SCORE_THRESHOLD):
    """Pick the largest face that exceeds the detection confidence threshold."""
    candidates = [f for f in faces if float(f.det_score) >= min_score]
    if not candidates:
        return None
    return max(candidates, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _extract_face_attributes(face) -> dict:
    """Pull age/gender predictions that InsightFace provides for free."""
    attrs = {}
    if hasattr(face, "age"):
        attrs["age"] = int(face.age)
    if hasattr(face, "gender"):
        attrs["gender"] = "M" if int(face.gender) == 1 else "F"
    return attrs


def _embed_single(app, img: np.ndarray):
    """Run detection + embedding on a single image, return (face, embedding) or (None, None)."""
    faces = app.get(img)
    face = _select_best_face(faces) if faces else None
    if face is None:
        return None, None
    return face, _normalize(face.embedding.copy())


def _tta_embed(app, img: np.ndarray) -> tuple:
    """Test-time augmentation: embed original + flip + brightness variants, average the vectors.

    Returns (face_from_original, averaged_embedding) or (None, None).
    """
    face, emb_orig = _embed_single(app, img)
    if emb_orig is None:
        return None, None

    embeddings = [emb_orig]

    flipped = cv2.flip(img, 1)
    _, emb_flip = _embed_single(app, flipped)
    if emb_flip is not None:
        embeddings.append(emb_flip)

    for delta in TTA_BRIGHTNESS_DELTAS:
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
        _, emb_b = _embed_single(app, bright)
        if emb_b is not None:
            embeddings.append(emb_b)

    avg = np.mean(embeddings, axis=0)
    return face, _normalize(avg)


def get_embedding(image_path: str, use_tta: bool = True) -> np.ndarray | None:
    """Extract the 512-d normalized face embedding from the best face in the image.

    Uses preprocessing (EXIF orient, upscale, CLAHE, sharpen) and optionally
    test-time augmentation for a more robust descriptor.
    """
    from utils.image_utils import load_cv2_image, preprocess_for_embedding

    img = load_cv2_image(image_path)
    img = preprocess_for_embedding(img)
    return get_embedding_from_array(img, use_tta=use_tta)


def get_embedding_from_array(img: np.ndarray, use_tta: bool = True) -> np.ndarray | None:
    """Extract embedding from a BGR numpy array (OpenCV format)."""
    app = _get_face_app()
    if use_tta:
        _, emb = _tta_embed(app, img)
    else:
        _, emb = _embed_single(app, img)
    return emb


def get_face_info(image_path: str) -> dict | None:
    """Get embedding + age/gender attributes for the best face in an image.

    Returns dict with keys: embedding, age, gender (or None if no face found).
    """
    from utils.image_utils import load_cv2_image, preprocess_for_embedding

    img = load_cv2_image(image_path)
    img = preprocess_for_embedding(img)
    app = _get_face_app()
    face, emb = _tta_embed(app, img)
    if emb is None:
        return None
    result = {"embedding": emb}
    result.update(_extract_face_attributes(face))
    return result


def get_all_embeddings(image_path: str) -> list[dict]:
    """Return embeddings for every detected face with bounding box info."""
    from utils.image_utils import load_cv2_image, preprocess_for_embedding

    img = load_cv2_image(image_path)
    img = preprocess_for_embedding(img)
    app = _get_face_app()
    faces = app.get(img)
    results = []
    for face in faces:
        if float(face.det_score) < DET_SCORE_THRESHOLD:
            continue
        emb = _normalize(face.embedding.copy())
        entry = {
            "embedding": emb,
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
        }
        entry.update(_extract_face_attributes(face))
        results.append(entry)
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def _augmented_embeddings(app, img: np.ndarray) -> list[np.ndarray]:
    """Generate embedding variants for index-time augmentation: original TTA + flipped TTA."""
    _, emb_orig = _tta_embed(app, img)
    if emb_orig is None:
        return []

    variants = [emb_orig]

    flipped = cv2.flip(img, 1)
    _, emb_flip = _tta_embed(app, flipped)
    if emb_flip is not None:
        variants.append(emb_flip)

    return variants


def batch_embed_folder(
    folder: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    augment: bool = True,
    age_variants: bool = True,
) -> list[dict]:
    """Embed all images in a folder. Returns list of {path, embedding, age?, gender?}.

    When augment=True, each image produces multiple embedding variants (original + flip)
    for better recall. All variants share the same path and metadata.

    When age_variants=True, synthetic aged/de-aged versions are also embedded
    for cross-age matching.
    """
    import os
    from utils.image_utils import SUPPORTED_EXTENSIONS, load_cv2_image, preprocess_for_embedding

    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    app = _get_face_app()
    results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(i, total, path)
        try:
            img = load_cv2_image(path)
            img = preprocess_for_embedding(img)

            faces = app.get(img)
            face = _select_best_face(faces) if faces else None
            if face is None:
                continue

            attrs = _extract_face_attributes(face)

            if augment:
                variants = _augmented_embeddings(app, img)
            else:
                _, emb = _tta_embed(app, img)
                variants = [emb] if emb is not None else []

            for emb in variants:
                entry = {"path": path, "embedding": emb}
                entry.update(attrs)
                results.append(entry)

            if age_variants:
                from core.age_transform import generate_age_variants
                source_age = attrs.get("age")
                for aged_img in generate_age_variants(img, source_age=source_age):
                    _, aged_emb = _tta_embed(app, aged_img)
                    if aged_emb is not None:
                        entry = {"path": path, "embedding": aged_emb}
                        entry.update(attrs)
                        results.append(entry)
        except Exception as e:
            log.warning("Failed to embed %s: %s", path, e)
    return results
