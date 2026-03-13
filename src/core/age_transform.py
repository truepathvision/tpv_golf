"""Optional face age transformation for generating synthetic age variants.

Uses the FRAN (Face Re-Aging Network) U-Net model when weights are available,
with a lightweight OpenCV fallback that simulates aging via texture/wrinkle
synthesis and skin tone shifts to produce useful embedding variants even
without the full model.
"""

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)

_fran_model = None
_fran_available: bool | None = None

FRAN_WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), ".tpv_golf", "models")
FRAN_WEIGHTS_FILE = os.path.join(FRAN_WEIGHTS_DIR, "best_unet_model.pth")
FRAN_HF_REPO = "timroelofs123/face_re-aging"


def is_fran_available() -> bool:
    """Check if FRAN model weights are downloaded and torch is installed."""
    global _fran_available
    if _fran_available is not None:
        return _fran_available
    try:
        import torch  # noqa: F401
        _fran_available = os.path.isfile(FRAN_WEIGHTS_FILE)
    except ImportError:
        _fran_available = False
    return _fran_available


def download_fran_weights() -> bool:
    """Download FRAN weights from Hugging Face. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(FRAN_WEIGHTS_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=FRAN_HF_REPO,
            filename="best_unet_model.pth",
            local_dir=FRAN_WEIGHTS_DIR,
        )
        global _fran_available
        _fran_available = True
        log.info("Downloaded FRAN weights to %s", FRAN_WEIGHTS_FILE)
        return True
    except Exception as e:
        log.warning("Failed to download FRAN weights: %s", e)
        return False


def _cv2_age_forward(img: np.ndarray, delta_years: int) -> np.ndarray:
    """Lightweight OpenCV-based age simulation.

    Applies subtle texture and tone changes that shift the embedding
    in an age-correlated direction without needing a neural network.

    Positive delta = age forward (add texture, slight warmth).
    Negative delta = de-age (smooth, slight coolness).
    """
    result = img.astype(np.float32)
    strength = abs(delta_years) / 30.0
    strength = min(strength, 1.0)

    if delta_years > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detail = cv2.Laplacian(gray, cv2.CV_32F)
        detail = np.clip(detail * strength * 0.3, -30, 30)
        for c in range(3):
            result[:, :, c] += detail

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] -= strength * 5
        lab[:, :, 1] += strength * 3
        result = cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        ).astype(np.float32)
    else:
        ksize = int(3 + strength * 4) | 1
        smoothed = cv2.GaussianBlur(img, (ksize, ksize), 0)
        alpha = strength * 0.4
        result = result * (1 - alpha) + smoothed.astype(np.float32) * alpha

        lab = cv2.cvtColor(
            np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        lab[:, :, 0] += strength * 4
        lab[:, :, 2] -= strength * 2
        result = cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        ).astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def _fran_transform(img: np.ndarray, source_age: int, target_age: int) -> np.ndarray:
    """Apply FRAN neural re-aging. Requires torch and downloaded weights."""
    import torch
    import torch.nn.functional as F

    global _fran_model
    if _fran_model is None:
        from core._fran_unet import FRANUNet
        _fran_model = FRANUNet()
        state = torch.load(FRAN_WEIGHTS_FILE, map_location="cpu", weights_only=True)
        _fran_model.load_state_dict(state)
        _fran_model.eval()
        log.info("Loaded FRAN U-Net model")

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)

    age_in = torch.tensor([[source_age / 100.0]])
    age_out = torch.tensor([[target_age / 100.0]])

    with torch.no_grad():
        out = _fran_model(tensor, age_in, age_out)

    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    out = (out * 255).astype(np.uint8)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def generate_age_variants(
    img: np.ndarray, source_age: int | None = None, deltas: list[int] | None = None,
) -> list[np.ndarray]:
    """Generate age-transformed variants of a face image.

    Returns a list of transformed BGR images (does NOT include the original).
    Uses FRAN if available, otherwise falls back to CV2 simulation.
    """
    if deltas is None:
        deltas = [-15, 15]

    variants = []
    use_fran = is_fran_available() and source_age is not None

    for delta in deltas:
        try:
            if use_fran:
                target_age = max(5, min(80, source_age + delta))
                variant = _fran_transform(img, source_age, target_age)
            else:
                variant = _cv2_age_forward(img, delta)
            variants.append(variant)
        except Exception as e:
            log.warning("Age transform (delta=%+d) failed: %s", delta, e)

    return variants
