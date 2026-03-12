from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

from ui.widgets.carousel_widget import CarouselWidget


class ResultsPanel(QWidget):
    """Right 2/3 panel: shows top-10 match results in a carousel."""

    match_selected = Signal(int, str, float)  # (index, path, score)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._header = QLabel("Search Results")
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")

        self._info_label = QLabel("Open an image and click Search to find matches")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #888; font-size: 13px; padding: 4px;")

        self._carousel = CarouselWidget()
        self._carousel.card_clicked.connect(self._on_card_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._header)
        layout.addWidget(self._info_label)
        layout.addWidget(self._carousel, 1)

    def set_results(self, results: list[dict]):
        """Display search results. Each dict has path, score, index_id."""
        count = len(results)
        self._info_label.setText(
            f"{count} match{'es' if count != 1 else ''} found — click a result to compare"
        )
        self._carousel.set_results(results)

    def clear(self):
        self._info_label.setText("Open an image and click Search to find matches")
        self._carousel.clear()

    def _on_card_clicked(self, index: int, path: str, score: float):
        self.match_selected.emit(index, path, score)
