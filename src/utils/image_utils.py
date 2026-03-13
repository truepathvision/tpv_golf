import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

MIN_FACE_RESOLUTION = 640


def is_supported_image(path: str) -> bool:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return f".{ext}" in SUPPORTED_EXTENSIONS


def load_cv2_image(path: str) -> np.ndarray:
    """Load image with EXIF auto-orient via Pillow, return as BGR numpy array."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_for_embedding(img: np.ndarray) -> np.ndarray:
    """Prepare a BGR image for face detection and embedding extraction.

    Steps: upscale small images, apply CLAHE, mild sharpening.
    """
    h, w = img.shape[:2]
    if max(h, w) < MIN_FACE_RESOLUTION:
        scale = MIN_FACE_RESOLUTION / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.3, blur, -0.3, 0)

    return img


def cv2_to_qpixmap(img: np.ndarray) -> QPixmap:
    if len(img.shape) == 3:
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    else:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def load_qpixmap(path: str) -> QPixmap:
    pixmap = QPixmap(path)
    if pixmap.isNull():
        raise ValueError(f"Could not load image: {path}")
    return pixmap


def generate_thumbnail(path: str, size: int = 200) -> QPixmap:
    pixmap = load_qpixmap(path)
    return pixmap.scaled(
        size, size,
        aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
        mode=Qt.TransformationMode.SmoothTransformation,
    )
