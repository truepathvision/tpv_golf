import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QAbstractItemView, QMessageBox,
    QFileDialog, QProgressDialog, QGroupBox, QFrame,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QSize
from PySide6.QtGui import QPixmap, QIcon

from core.vector_store import VectorStore

THUMB_SIZE = 64


class _AddFolderWorker(QObject):
    """Background worker that embeds all images in a folder and adds them to the store."""
    progress = Signal(int, int, str)
    finished = Signal(int)  # count added
    error = Signal(str)

    def __init__(self, store: VectorStore, folder: str):
        super().__init__()
        self._store = store
        self._folder = folder
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from core.embedder import batch_embed_folder
            items = batch_embed_folder(
                self._folder,
                progress_callback=lambda cur, tot, p: self.progress.emit(cur, tot, p),
                cancel_check=lambda: self._cancelled,
            )
            self._store.add_embeddings(items)
            self.finished.emit(len(items))
        except Exception as e:
            self.error.emit(str(e))


class _AddSingleWorker(QObject):
    """Background worker to embed and add a single image."""
    finished = Signal(bool, str)  # (success, path)
    error = Signal(str)

    def __init__(self, store: VectorStore, path: str):
        super().__init__()
        self._store = store
        self._path = path

    def run(self):
        try:
            ok = self._store.add_single_image(self._path)
            self.finished.emit(ok, self._path)
        except Exception as e:
            self.error.emit(str(e))


class AdminPanel(QDialog):
    """Admin panel for managing the target vector database: view, add, delete entries."""

    database_changed = Signal()

    def __init__(self, vector_store: VectorStore, parent=None):
        super().__init__(parent)
        self._store = vector_store
        self._worker_thread: QThread | None = None

        self.setWindowTitle("Admin — Target Database Manager")
        self.resize(1050, 700)

        # --- Stats group ---
        stats_group = QGroupBox("Database Statistics")
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet("font-size: 13px; padding: 6px;")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.addWidget(self._stats_label)

        # --- Table of entries ---
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Thumb", "#", "Image Path"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, THUMB_SIZE + 12)
        self._table.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setDefaultSectionSize(THUMB_SIZE + 8)

        # --- Action buttons ---
        self._add_image_btn = QPushButton("Add Image...")
        self._add_image_btn.setToolTip("Embed a single image and add to the database")
        self._add_image_btn.clicked.connect(self._on_add_image)

        self._add_folder_btn = QPushButton("Add Folder...")
        self._add_folder_btn.setToolTip("Embed all images in a folder and add to the database")
        self._add_folder_btn.clicked.connect(self._on_add_folder)

        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.setToolTip("Remove selected entries from the database")
        self._delete_btn.clicked.connect(self._on_delete)
        self._delete_btn.setStyleSheet(
            "QPushButton { color: #ff6b6b; } QPushButton:hover { border-color: #ff6b6b; }"
        )

        self._delete_all_btn = QPushButton("Clear Database")
        self._delete_all_btn.setToolTip("Remove all entries from the database")
        self._delete_all_btn.clicked.connect(self._on_delete_all)
        self._delete_all_btn.setStyleSheet(
            "QPushButton { color: #ff6b6b; } QPushButton:hover { border-color: #ff6b6b; }"
        )

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #555;")

        self._save_btn = QPushButton("Save DB...")
        self._save_btn.setToolTip("Save the current database to disk")
        self._save_btn.clicked.connect(self._on_save)

        self._load_btn = QPushButton("Load DB...")
        self._load_btn.setToolTip("Load a database from disk (replaces current)")
        self._load_btn.clicked.connect(self._on_load)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._add_image_btn)
        btn_row.addWidget(self._add_folder_btn)
        btn_row.addWidget(sep)
        btn_row.addWidget(self._delete_btn)
        btn_row.addWidget(self._delete_all_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._refresh_btn)

        # --- Status ---
        self._action_label = QLabel("")
        self._action_label.setStyleSheet("color: #4fc3f7; padding: 4px;")

        layout = QVBoxLayout(self)
        layout.addWidget(stats_group)
        layout.addWidget(self._table, 1)
        layout.addLayout(btn_row)
        layout.addWidget(self._action_label)

        self._refresh()

    # ---- Refresh ----

    def _refresh(self):
        self._table.setSortingEnabled(False)
        entries = self._store.get_all_entries()
        self._table.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            thumb_item = QTableWidgetItem()
            path = entry["path"]
            if os.path.isfile(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    pix = pix.scaled(
                        THUMB_SIZE, THUMB_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    thumb_item.setIcon(QIcon(pix))
            thumb_item.setData(Qt.ItemDataRole.UserRole, entry["index_id"])
            self._table.setItem(i, 0, thumb_item)

            idx_item = QTableWidgetItem(str(entry["index_id"]))
            idx_item.setData(Qt.ItemDataRole.UserRole, entry["index_id"])
            self._table.setItem(i, 1, idx_item)

            path_item = QTableWidgetItem(path)
            path_item.setToolTip(path)
            self._table.setItem(i, 2, path_item)

        self._table.setSortingEnabled(True)

        count = self._store.count()
        self._stats_label.setText(
            f"Total vectors: {count}    |    "
            f"Index type: FAISS IndexFlatIP (cosine similarity)    |    "
            f"Dimension: 512"
        )

    def _set_action(self, text: str):
        self._action_label.setText(text)

    # ---- Add single image ----

    def _on_add_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images to Add", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;All Files (*)",
        )
        if not paths:
            return

        self._set_buttons_enabled(False)
        self._set_action(f"Embedding {len(paths)} image(s)...")
        self._pending_paths = list(paths)
        self._added_count = 0
        self._failed_count = 0
        self._process_next_single()

    def _process_next_single(self):
        if not self._pending_paths:
            self._set_buttons_enabled(True)
            self._set_action(
                f"Done: {self._added_count} added, {self._failed_count} failed (no face detected)"
            )
            self._refresh()
            self.database_changed.emit()
            return

        path = self._pending_paths.pop(0)
        thread = QThread()
        worker = _AddSingleWorker(self._store, path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(
            self._on_single_done, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(
            self._on_single_error, Qt.ConnectionType.QueuedConnection
        )

        self._worker_thread = thread
        self._worker_ref = worker
        thread.start()

    def _on_single_done(self, ok: bool, path: str):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        if ok:
            self._added_count += 1
        else:
            self._failed_count += 1
        remaining = len(self._pending_paths)
        self._set_action(
            f"Embedding... ({self._added_count} added, {self._failed_count} failed, {remaining} remaining)"
        )
        self._process_next_single()

    def _on_single_error(self, msg: str):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._failed_count += 1
        self._process_next_single()

    # ---- Add folder ----

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder to Add")
        if not folder:
            return

        self._folder_progress = QProgressDialog("Adding images from folder...", "Cancel", 0, 0, self)
        self._folder_progress.setWindowTitle("Adding Folder")
        self._folder_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._folder_progress.setMinimumDuration(0)
        self._folder_progress.show()

        thread = QThread()
        worker = _AddFolderWorker(self._store, folder)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(
            self._on_folder_progress, Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(
            self._on_folder_done, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(
            self._on_folder_error, Qt.ConnectionType.QueuedConnection
        )
        self._folder_progress.canceled.connect(worker.cancel)

        self._worker_thread = thread
        self._worker_ref = worker
        thread.start()

    def _on_folder_progress(self, current: int, total: int, path: str):
        if self._folder_progress and total > 0:
            self._folder_progress.setMaximum(total)
            self._folder_progress.setValue(current)
            self._folder_progress.setLabelText(
                f"Processing {current + 1}/{total}\n{os.path.basename(path)}"
            )

    def _on_folder_done(self, count: int):
        if self._folder_progress:
            self._folder_progress.close()
            self._folder_progress = None
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._refresh()
        self.database_changed.emit()
        self._set_action(f"Added {count} images from folder")
        QMessageBox.information(self, "Folder Added", f"Added {count} face embeddings from folder.")

    def _on_folder_error(self, msg: str):
        if self._folder_progress:
            self._folder_progress.close()
            self._folder_progress = None
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        QMessageBox.critical(self, "Error", f"Failed to add folder:\n{msg}")

    # ---- Delete ----

    def _on_delete(self):
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Select rows to delete first.")
            return

        indices = set()
        for model_idx in selected_rows:
            row = model_idx.row()
            item = self._table.item(row, 1)
            if item:
                indices.add(item.data(Qt.ItemDataRole.UserRole))

        count = len(indices)
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Remove {count} entry/entries from the database?\n\nThis does not delete the image files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._store.remove_by_indices(indices)
        self._refresh()
        self.database_changed.emit()
        self._set_action(f"Deleted {count} entries")

    def _on_delete_all(self):
        if self._store.is_empty():
            return
        reply = QMessageBox.warning(
            self, "Clear Database",
            f"Remove ALL {self._store.count()} entries from the database?\n\n"
            "This cannot be undone (unless you reload from a saved file).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._store.remove_by_indices(set(range(self._store.count())))
        self._refresh()
        self.database_changed.emit()
        self._set_action("Database cleared")

    # ---- Save / Load ----

    def _on_save(self):
        if self._store.is_empty():
            QMessageBox.warning(self, "Empty", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Database", "vectors", "FAISS Index (*.faiss)"
        )
        if not path:
            return
        base = path.rsplit(".faiss", 1)[0]
        try:
            self._store.save(base)
            self._set_action(f"Saved to {base}.faiss")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Database", "", "FAISS Index (*.faiss);;All Files (*)"
        )
        if not path:
            return
        base = path.rsplit(".faiss", 1)[0]
        try:
            self._store.load(base)
            self._refresh()
            self.database_changed.emit()
            self._set_action(f"Loaded {self._store.count()} vectors from {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    # ---- Helpers ----

    def _set_buttons_enabled(self, enabled: bool):
        self._add_image_btn.setEnabled(enabled)
        self._add_folder_btn.setEnabled(enabled)
        self._delete_btn.setEnabled(enabled)
        self._delete_all_btn.setEnabled(enabled)
        self._save_btn.setEnabled(enabled)
        self._load_btn.setEnabled(enabled)
