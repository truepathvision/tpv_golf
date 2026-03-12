from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent

ZOOM_FACTOR = 1.15
MIN_ZOOM = 0.01
MAX_ZOOM = 100.0
SCENE_PADDING = 4.0  # multiplier: scene rect = image rect expanded by this factor


class PanZoomGraphicsView(QGraphicsView):
    """QGraphicsView with mouse-wheel zoom centered on cursor and click-drag pan."""

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._current_zoom = 1.0
        self._image_rect = QRectF()

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._image_rect = QRectF(pixmap.rect())
        self._update_scene_rect()
        self.resetTransform()
        self._current_zoom = 1.0
        self.fit_in_view()

    def _update_scene_rect(self):
        r = self._image_rect
        pad_w = r.width() * SCENE_PADDING
        pad_h = r.height() * SCENE_PADDING
        self._scene.setSceneRect(r.adjusted(-pad_w, -pad_h, pad_w, pad_h))

    def clear_image(self):
        self._scene.clear()
        self._pixmap_item = None
        self._image_rect = QRectF()
        self._current_zoom = 1.0
        self.zoom_changed.emit(self._current_zoom)

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def fit_in_view(self):
        if not self._pixmap_item:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_zoom = self.transform().m11()
        self.zoom_changed.emit(self._current_zoom)

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap_item:
            return

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            factor = ZOOM_FACTOR
        else:
            factor = 1.0 / ZOOM_FACTOR

        new_zoom = self._current_zoom * factor
        if new_zoom < MIN_ZOOM or new_zoom > MAX_ZOOM:
            return

        self.scale(factor, factor)
        self._current_zoom = new_zoom

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self.zoom_changed.emit(self._current_zoom)

    def get_zoom_percent(self) -> float:
        return self._current_zoom * 100.0


class ImageViewer(QWidget):
    """Full image viewer widget with toolbar: fit-to-view button and zoom indicator."""

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = PanZoomGraphicsView(self)
        self._view.zoom_changed.connect(self._on_zoom_changed)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(60)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setFixedWidth(40)
        self._fit_btn.clicked.connect(self._view.fit_in_view)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)
        toolbar.addWidget(self._fit_btn)
        toolbar.addStretch()
        toolbar.addWidget(self._zoom_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, 1)
        layout.addLayout(toolbar)

    def set_image(self, pixmap: QPixmap):
        self._view.set_pixmap(pixmap)

    def set_image_from_path(self, path: str):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._view.set_pixmap(pixmap)

    def clear_image(self):
        self._view.clear_image()
        self._zoom_label.setText("---")

    def has_image(self) -> bool:
        return self._view.has_image()

    def fit_in_view(self):
        self._view.fit_in_view()

    @property
    def view(self) -> PanZoomGraphicsView:
        return self._view

    def _on_zoom_changed(self, zoom: float):
        self._zoom_label.setText(f"{zoom * 100:.0f}%")
        self.zoom_changed.emit(zoom)
