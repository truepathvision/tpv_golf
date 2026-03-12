import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QSizePolicy, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
import numpy as np

from ui.widgets.image_viewer import ImageViewer


class _EmbedWorker(QObject):
    finished = Signal(object)  # np.ndarray or None
    error = Signal(str)
    status = Signal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self._path = image_path

    def run(self):
        try:
            self.status.emit("Loading model...")
            from core.embedder import get_embedding
            self.status.emit("Computing embedding...")
            emb = get_embedding(self._path)
            self.finished.emit(emb)
        except Exception as e:
            self.error.emit(str(e))


class SourcePanel(QWidget):
    """Left 1/3 panel: open image, view it, search for matches."""

    search_requested = Signal(np.ndarray, str)  # (embedding, image_path)
    image_opened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_path: str | None = None
        self._current_embedding: np.ndarray | None = None
        self._worker_thread: QThread | None = None

        self._viewer = ImageViewer(self)

        self._open_btn = QPushButton("Open Image")
        self._open_btn.clicked.connect(self._on_open)

        self._search_btn = QPushButton("Search Matches")
        self._search_btn.setEnabled(False)
        self._search_btn.clicked.connect(self._on_search)

        self._save_emb_btn = QPushButton("Save Embedding")
        self._save_emb_btn.setEnabled(False)
        self._save_emb_btn.setToolTip("Save the probe image's embedding vector to disk")
        self._save_emb_btn.clicked.connect(self._on_save_embedding)

        self._status_label = QLabel("No image loaded")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()

        self._filename_label = QLabel("")
        self._filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._filename_label.setStyleSheet("font-weight: bold;")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self._open_btn)
        btn_layout.addWidget(self._search_btn)
        btn_layout.addWidget(self._save_emb_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(btn_layout)
        layout.addWidget(self._filename_label)
        layout.addWidget(self._viewer, 1)
        layout.addWidget(self._progress)
        layout.addWidget(self._status_label)

    @property
    def current_path(self) -> str | None:
        return self._current_path

    @property
    def current_embedding(self) -> np.ndarray | None:
        return self._current_embedding

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;All Files (*)",
        )
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str):
        self._current_path = path
        self._current_embedding = None
        self._search_btn.setEnabled(False)
        self._save_emb_btn.setEnabled(False)

        self._viewer.set_image_from_path(path)
        self._filename_label.setText(os.path.basename(path))
        self._status_label.setText("Embedding image...")
        self._progress.show()

        self.image_opened.emit(path)
        self._start_embed(path)

    def _start_embed(self, path: str):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()

        self._worker_thread = QThread()
        worker = _EmbedWorker(path)
        worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(worker.run)

        # All slots must run on the main thread to avoid segfaults.
        # QueuedConnection ensures signals emitted from the worker thread
        # are dispatched via the main thread's event loop.
        worker.finished.connect(
            self._on_embed_done, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(
            self._on_embed_error, Qt.ConnectionType.QueuedConnection
        )
        worker.status.connect(
            self._on_embed_status, Qt.ConnectionType.QueuedConnection
        )

        self._worker_ref = worker
        self._worker_thread.start()

    def _on_embed_status(self, text: str):
        self._status_label.setText(text)

    def _on_embed_done(self, embedding):
        self._progress.hide()
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        if embedding is not None:
            self._current_embedding = embedding
            self._search_btn.setEnabled(True)
            self._save_emb_btn.setEnabled(True)
            self._status_label.setText("Ready — click Search Matches")
        else:
            self._status_label.setText("No face detected in image")

    def _on_embed_error(self, msg: str):
        self._progress.hide()
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._status_label.setText(f"Error: {msg}")

    def _on_save_embedding(self):
        if self._current_embedding is None or self._current_path is None:
            return

        base_name = os.path.splitext(os.path.basename(self._current_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Embedding Vector", base_name + "_embedding",
            "NumPy Array (*.npy);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            if path.endswith(".json"):
                data = {
                    "source_image": self._current_path,
                    "embedding": self._current_embedding.tolist(),
                    "dimension": len(self._current_embedding),
                }
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                if not path.endswith(".npy"):
                    path += ".npy"
                np.save(path, self._current_embedding)

            self._status_label.setText(f"Embedding saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save embedding:\n{e}")

    def _on_search(self):
        if self._current_embedding is not None and self._current_path:
            self.search_requested.emit(self._current_embedding, self._current_path)
