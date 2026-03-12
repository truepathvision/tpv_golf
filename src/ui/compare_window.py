import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QCheckBox, QTextEdit, QSplitter, QFrame,
)
from PySide6.QtCore import Qt, Signal, QTimer

from ui.widgets.image_viewer import ImageViewer


class CompareWindow(QMainWindow):
    """Side-by-side image comparison window with synced pan/zoom."""

    note_added = Signal(str)  # note_text

    def __init__(
        self,
        source_path: str,
        match_path: str,
        score: float,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = source_path
        self._match_path = match_path
        self._score = score
        self._sync_enabled = True

        self.setWindowTitle(
            f"Compare — {os.path.basename(source_path)} vs {os.path.basename(match_path)}"
        )
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)

        # --- Left viewer (source) ---
        self._source_viewer = ImageViewer()
        self._source_viewer.set_image_from_path(source_path)

        source_label = QLabel(f"Source: {os.path.basename(source_path)}")
        source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(source_label)
        left_layout.addWidget(self._source_viewer, 1)

        # --- Right viewer (match) ---
        self._match_viewer = ImageViewer()
        self._match_viewer.set_image_from_path(match_path)

        match_label = QLabel(f"Match: {os.path.basename(match_path)}  |  Score: {score:.4f}")
        match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        match_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px; color: #4fc3f7;")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(match_label)
        right_layout.addWidget(self._match_viewer, 1)

        # --- Splitter for viewers ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 700])

        # --- Bottom toolbar ---
        self._sync_check = QCheckBox("Sync Pan/Zoom")
        self._sync_check.setChecked(True)
        self._sync_check.toggled.connect(self._on_sync_toggled)

        self._fit_both_btn = QPushButton("Fit Both")
        self._fit_both_btn.clicked.connect(self._fit_both)

        self._note_edit = QTextEdit()
        self._note_edit.setPlaceholderText("Add a note about this comparison...")
        self._note_edit.setMaximumHeight(60)

        self._save_note_btn = QPushButton("Save Note")
        self._save_note_btn.clicked.connect(self._on_save_note)

        note_row = QHBoxLayout()
        note_row.addWidget(self._note_edit, 1)
        note_row.addWidget(self._save_note_btn)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._sync_check)
        toolbar.addWidget(self._fit_both_btn)
        toolbar.addStretch()

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(splitter, 1)
        main_layout.addLayout(toolbar)
        main_layout.addLayout(note_row)

        self._connect_sync()
        QTimer.singleShot(0, self._fit_both)

    def _connect_sync(self):
        src_view = self._source_viewer.view
        match_view = self._match_viewer.view

        src_view.horizontalScrollBar().valueChanged.connect(self._sync_h_from_source)
        src_view.verticalScrollBar().valueChanged.connect(self._sync_v_from_source)
        match_view.horizontalScrollBar().valueChanged.connect(self._sync_h_from_match)
        match_view.verticalScrollBar().valueChanged.connect(self._sync_v_from_match)

        src_view.zoom_changed.connect(self._sync_zoom_from_source)
        match_view.zoom_changed.connect(self._sync_zoom_from_match)

    def _on_sync_toggled(self, enabled: bool):
        self._sync_enabled = enabled

    def _sync_h_from_source(self, val):
        if self._sync_enabled:
            self._match_viewer.view.horizontalScrollBar().blockSignals(True)
            self._match_viewer.view.horizontalScrollBar().setValue(val)
            self._match_viewer.view.horizontalScrollBar().blockSignals(False)

    def _sync_v_from_source(self, val):
        if self._sync_enabled:
            self._match_viewer.view.verticalScrollBar().blockSignals(True)
            self._match_viewer.view.verticalScrollBar().setValue(val)
            self._match_viewer.view.verticalScrollBar().blockSignals(False)

    def _sync_h_from_match(self, val):
        if self._sync_enabled:
            self._source_viewer.view.horizontalScrollBar().blockSignals(True)
            self._source_viewer.view.horizontalScrollBar().setValue(val)
            self._source_viewer.view.horizontalScrollBar().blockSignals(False)

    def _sync_v_from_match(self, val):
        if self._sync_enabled:
            self._source_viewer.view.verticalScrollBar().blockSignals(True)
            self._source_viewer.view.verticalScrollBar().setValue(val)
            self._source_viewer.view.verticalScrollBar().blockSignals(False)

    def _sync_zoom_from_source(self, zoom: float):
        if not self._sync_enabled:
            return
        match_view = self._match_viewer.view
        current = match_view._current_zoom
        if abs(current - zoom) > 0.001:
            factor = zoom / current if current > 0 else 1.0
            match_view.blockSignals(True)
            match_view.scale(factor, factor)
            match_view._current_zoom = zoom
            match_view.blockSignals(False)

    def _sync_zoom_from_match(self, zoom: float):
        if not self._sync_enabled:
            return
        src_view = self._source_viewer.view
        current = src_view._current_zoom
        if abs(current - zoom) > 0.001:
            factor = zoom / current if current > 0 else 1.0
            src_view.blockSignals(True)
            src_view.scale(factor, factor)
            src_view._current_zoom = zoom
            src_view.blockSignals(False)

    def _fit_both(self):
        self._source_viewer.fit_in_view()
        self._match_viewer.fit_in_view()

    def _on_save_note(self):
        text = self._note_edit.toPlainText().strip()
        if text:
            self.note_added.emit(text)
            self._note_edit.clear()
