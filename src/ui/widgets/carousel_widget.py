import os

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QCursor


class CarouselCard(QFrame):
    """Single thumbnail card in the carousel."""

    clicked = Signal(int)  # index

    def __init__(self, index: int, image_path: str, score: float, parent=None):
        super().__init__(parent)
        self._index = index
        self._image_path = image_path
        self._score = score

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(220, 280)
        self.setStyleSheet("""
            CarouselCard {
                border: 2px solid #555;
                border-radius: 6px;
                background: #2b2b2b;
            }
            CarouselCard:hover {
                border-color: #4fc3f7;
                background: #333;
            }
        """)

        thumb = QPixmap(image_path)
        if not thumb.isNull():
            thumb = thumb.scaled(
                200, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        thumb_label = QLabel()
        thumb_label.setPixmap(thumb)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setFixedHeight(210)
        thumb_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        name_label = QLabel(os.path.basename(image_path))
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("color: #ddd; font-size: 11px;")
        name_label.setWordWrap(True)
        name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        score_label = QLabel(f"Score: {score:.4f}")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 12px;")
        score_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        rank_label = QLabel(f"#{index + 1}")
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rank_label.setStyleSheet("color: #aaa; font-size: 10px;")
        rank_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)
        layout.addWidget(rank_label)
        layout.addWidget(thumb_label, 1)
        layout.addWidget(score_label)
        layout.addWidget(name_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)


class CarouselWidget(QWidget):
    """Horizontal scrollable carousel of match thumbnail cards."""

    card_clicked = Signal(int, str, float)  # (index, path, score)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[dict] = []

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._container_layout = QHBoxLayout(self._container)
        self._container_layout.setContentsMargins(8, 8, 8, 8)
        self._container_layout.setSpacing(12)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)

        self._left_btn = QPushButton("<")
        self._left_btn.setFixedSize(30, 60)
        self._left_btn.clicked.connect(self._scroll_left)

        self._right_btn = QPushButton(">")
        self._right_btn.setFixedSize(30, 60)
        self._right_btn.clicked.connect(self._scroll_right)

        nav_layout = QHBoxLayout(self)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.addWidget(self._left_btn)
        nav_layout.addWidget(self._scroll, 1)
        nav_layout.addWidget(self._right_btn)

    def set_results(self, results: list[dict]):
        """Set results: list of {path, score, index_id}."""
        self._items = results
        self._rebuild()

    def clear(self):
        self._items = []
        self._rebuild()

    def _rebuild(self):
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, item in enumerate(self._items):
            card = CarouselCard(i, item["path"], item["score"])
            card.clicked.connect(lambda idx=i: self._on_card_clicked(idx))
            self._container_layout.addWidget(card)
        self._container_layout.addStretch()

    def _on_card_clicked(self, index: int):
        if 0 <= index < len(self._items):
            item = self._items[index]
            self.card_clicked.emit(index, item["path"], item["score"])

    def _scroll_left(self):
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(bar.value() - 240)

    def _scroll_right(self):
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(bar.value() + 240)
