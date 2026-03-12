import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported_image(path: str) -> bool:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return f".{ext}" in SUPPORTED_EXTENSIONS


def load_cv2_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
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
