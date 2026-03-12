import logging
import os

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QMenuBar, QMenu, QToolBar,
    QStatusBar, QFileDialog, QProgressDialog, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QAction

from core.case_store import CaseStore
from core.vector_store import VectorStore, DEFAULT_STORE_PATH
from ui.source_panel import SourcePanel
from ui.results_panel import ResultsPanel
from ui.compare_window import CompareWindow
from ui.case_manager_dialog import CaseManagerDialog
from ui.admin_panel import AdminPanel

log = logging.getLogger(__name__)


class _BuildDBWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal()
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
            self._store.build_from_folder(
                self._folder,
                progress_callback=lambda cur, tot, p: self.progress.emit(cur, tot, p),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPV_Golf — Image Matcher")
        self.resize(1600, 900)

        self._case_store = CaseStore()
        self._vector_store = VectorStore()
        self._active_case_id: int | None = None
        self._compare_windows: list[CompareWindow] = []
        self._build_thread: QThread | None = None

        self._source_panel = SourcePanel()
        self._results_panel = ResultsPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._source_panel)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        self._apply_dark_theme()
        self._auto_load_db()

    # ---- Menu bar ----

    def _setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self._act_open = QAction("Open Image", self)
        self._act_open.setShortcut("Ctrl+O")
        self._act_open.triggered.connect(self._source_panel._on_open)
        file_menu.addAction(self._act_open)

        file_menu.addSeparator()

        self._act_build_db = QAction("Build Vector Database...", self)
        self._act_build_db.triggered.connect(self._on_build_db)
        file_menu.addAction(self._act_build_db)

        self._act_load_db = QAction("Load Vector Database...", self)
        self._act_load_db.triggered.connect(self._on_load_db)
        file_menu.addAction(self._act_load_db)

        self._act_save_db = QAction("Save Vector Database...", self)
        self._act_save_db.triggered.connect(self._on_save_db)
        file_menu.addAction(self._act_save_db)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        case_menu = menubar.addMenu("Case")
        self._act_new_case = QAction("New Case", self)
        self._act_new_case.setShortcut("Ctrl+N")
        self._act_new_case.triggered.connect(self._on_new_case)
        case_menu.addAction(self._act_new_case)

        self._act_manage_cases = QAction("Manage Cases...", self)
        self._act_manage_cases.triggered.connect(self._on_manage_cases)
        case_menu.addAction(self._act_manage_cases)

        case_menu.addSeparator()

        self._act_close_case = QAction("Close Active Case", self)
        self._act_close_case.triggered.connect(self._on_close_active_case)
        case_menu.addAction(self._act_close_case)

        admin_menu = menubar.addMenu("Admin")
        self._act_admin = QAction("Manage Database...", self)
        self._act_admin.setShortcut("Ctrl+D")
        self._act_admin.triggered.connect(self._on_admin_panel)
        admin_menu.addAction(self._act_admin)

        help_menu = menubar.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ---- Toolbar ----

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self._act_open)
        toolbar.addAction(self._act_build_db)
        toolbar.addSeparator()
        toolbar.addAction(self._act_new_case)
        toolbar.addAction(self._act_manage_cases)
        toolbar.addSeparator()
        toolbar.addAction(self._act_admin)

    # ---- Status bar ----

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        self._case_label = QLabel("No active case")
        self._db_label = QLabel("DB: empty")
        self._statusbar.addWidget(self._case_label, 1)
        self._statusbar.addPermanentWidget(self._db_label)
        self._update_status()

    def _update_status(self):
        if self._active_case_id:
            try:
                case = self._case_store.get_case(self._active_case_id)
                self._case_label.setText(f"Case: {case.name} (#{case.id})")
            except ValueError:
                self._case_label.setText("No active case")
                self._active_case_id = None
        else:
            self._case_label.setText("No active case")

        count = self._vector_store.count()
        self._db_label.setText(f"DB: {count} vectors" if count else "DB: empty")

    # ---- Signal wiring ----

    def _connect_signals(self):
        self._source_panel.search_requested.connect(self._on_search)
        self._source_panel.image_opened.connect(self._on_image_opened)
        self._results_panel.match_selected.connect(self._on_match_selected)

    def _on_image_opened(self, path: str):
        if self._active_case_id:
            self._case_store.log_action(
                self._active_case_id, "open_image", {"path": path}
            )
            self._case_store.add_image(self._active_case_id, path, role="source")

    def _on_search(self, embedding: np.ndarray, source_path: str):
        if self._vector_store.is_empty():
            QMessageBox.warning(
                self, "No Database",
                "No vector database loaded. Use File > Build Vector Database first.",
            )
            return

        results = self._vector_store.search(embedding, k=10)
        result_dicts = [{"path": r.path, "score": r.score, "index_id": r.index_id} for r in results]
        self._results_panel.set_results(result_dicts)

        if self._active_case_id:
            self._case_store.log_action(
                self._active_case_id, "run_search",
                {"source": source_path, "matches": len(results)},
            )
            for r in results:
                self._case_store.add_image(
                    self._active_case_id, r.path, role="match"
                )

        self._statusbar.showMessage(f"Found {len(results)} matches", 5000)

    def _on_match_selected(self, index: int, path: str, score: float):
        source_path = self._source_panel.current_path
        if not source_path:
            return

        win = CompareWindow(source_path, path, score, parent=None)
        win.note_added.connect(lambda text: self._on_compare_note(text, path))
        win.show()
        self._compare_windows.append(win)

        if self._active_case_id:
            self._case_store.log_action(
                self._active_case_id, "compare",
                {"source": source_path, "match": path, "score": score},
            )

    def _on_compare_note(self, text: str, match_path: str):
        if self._active_case_id:
            self._case_store.add_note(self._active_case_id, text)
            self._case_store.log_action(
                self._active_case_id, "add_note",
                {"match_path": match_path, "note": text[:200]},
            )

    # ---- Build DB ----

    def _on_build_db(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        self._build_progress = QProgressDialog("Building vector database...", "Cancel", 0, 0, self)
        self._build_progress.setWindowTitle("Building Database")
        self._build_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._build_progress.setMinimumDuration(0)
        self._build_progress.show()

        thread = QThread()
        worker = _BuildDBWorker(self._vector_store, folder)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(
            self._on_build_progress, Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(
            self._on_build_done, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(
            self._on_build_error, Qt.ConnectionType.QueuedConnection
        )
        self._build_progress.canceled.connect(worker.cancel)

        self._build_thread = thread
        self._build_worker_ref = worker
        thread.start()

    def _on_build_progress(self, current: int, total: int, path: str):
        if self._build_progress and total > 0:
            self._build_progress.setMaximum(total)
            self._build_progress.setValue(current)
            self._build_progress.setLabelText(
                f"Processing {current + 1}/{total}\n{os.path.basename(path)}"
            )

    def _on_build_done(self):
        if self._build_progress:
            self._build_progress.close()
            self._build_progress = None
        if self._build_thread and self._build_thread.isRunning():
            self._build_thread.quit()
            self._build_thread.wait()
        count = self._vector_store.count()
        self._update_status()
        self._auto_save_db()
        QMessageBox.information(
            self, "Database Built",
            f"Vector database built with {count} face embeddings.",
        )
        if self._active_case_id:
            self._case_store.log_action(
                self._active_case_id, "build_db", {"count": count}
            )

    def _on_build_error(self, msg: str):
        if self._build_progress:
            self._build_progress.close()
            self._build_progress = None
        if self._build_thread and self._build_thread.isRunning():
            self._build_thread.quit()
            self._build_thread.wait()
        QMessageBox.critical(self, "Build Error", f"Failed to build database:\n{msg}")

    def _on_load_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Vector Database", "", "FAISS Index (*.faiss);;All Files (*)"
        )
        if not path:
            return
        base = path.rsplit(".faiss", 1)[0]
        try:
            self._vector_store.load(base)
            self._update_status()
            QMessageBox.information(
                self, "Database Loaded",
                f"Loaded {self._vector_store.count()} vectors.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _on_save_db(self):
        if self._vector_store.is_empty():
            QMessageBox.warning(self, "No Database", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Vector Database", "vectors", "FAISS Index (*.faiss)"
        )
        if not path:
            return
        base = path.rsplit(".faiss", 1)[0]
        try:
            self._vector_store.save(base)
            QMessageBox.information(self, "Saved", "Vector database saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ---- Case management ----

    def _on_new_case(self):
        from ui.case_manager_dialog import NewCaseDialog
        dlg = NewCaseDialog(self)
        if dlg.exec() == NewCaseDialog.DialogCode.Accepted and dlg.case_name:
            case = self._case_store.create_case(dlg.case_name, dlg.case_description)
            self._case_store.log_action(case.id, "create_case", {"name": case.name})
            self._active_case_id = case.id
            self._update_status()
            self._statusbar.showMessage(f"Case '{case.name}' created and activated", 5000)

    def _on_manage_cases(self):
        dlg = CaseManagerDialog(self._case_store, self._active_case_id, self)
        dlg.case_opened.connect(self._activate_case)
        dlg.case_created.connect(self._activate_case)
        dlg.exec()
        self._update_status()

    def _activate_case(self, case_id: int):
        self._active_case_id = case_id
        self._case_store.log_action(case_id, "open_case")
        self._update_status()

    def _on_close_active_case(self):
        if self._active_case_id:
            self._case_store.close_case(self._active_case_id)
            self._statusbar.showMessage("Case closed", 3000)
            self._active_case_id = None
            self._update_status()

    # ---- Admin panel ----

    def _on_admin_panel(self):
        dlg = AdminPanel(self._vector_store, self)
        dlg.database_changed.connect(self._update_status)
        dlg.database_changed.connect(self._auto_save_db)
        if self._active_case_id:
            dlg.database_changed.connect(
                lambda: self._case_store.log_action(
                    self._active_case_id, "admin_db_change",
                    {"count": self._vector_store.count()},
                )
            )
        dlg.exec()
        self._update_status()

    def _on_about(self):
        QMessageBox.about(
            self, "About TPV_Golf",
            "TPV_Golf — Image Matcher\n\n"
            "Face image matching using InsightFace embeddings\n"
            "and FAISS vector search.\n\n"
            "Built with PySide6.",
        )

    # ---- Styling ----

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #dcdcdc;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #dcdcdc;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #dcdcdc;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
            QToolBar {
                background-color: #2d2d2d;
                border-bottom: 1px solid #555;
                spacing: 6px;
                padding: 2px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #dcdcdc;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
                border-color: #444;
            }
            QStatusBar {
                background-color: #252525;
                color: #aaa;
                border-top: 1px solid #555;
            }
            QSplitter::handle {
                background-color: #555;
                width: 3px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
                color: #dcdcdc;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QTableWidget {
                background-color: #252525;
                color: #dcdcdc;
                gridline-color: #444;
                border: 1px solid #555;
            }
            QTableWidget::item:selected {
                background-color: #3d6070;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #dcdcdc;
                border: 1px solid #444;
                padding: 4px;
            }
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                color: #dcdcdc;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #4fc3f7;
            }
            QLabel {
                color: #dcdcdc;
            }
            QDialog {
                background-color: #1e1e1e;
                color: #dcdcdc;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #dcdcdc;
                border: 1px solid #555;
                padding: 6px 16px;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
                border-bottom-color: #1e1e1e;
            }
            QCheckBox {
                color: #dcdcdc;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #4fc3f7;
                border-color: #4fc3f7;
            }
            QScrollBar:horizontal, QScrollBar:vertical {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 4px;
                min-width: 20px;
                min-height: 20px;
            }
            QScrollBar::handle:hover {
                background-color: #777;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0px;
                width: 0px;
            }
        """)

    def _auto_load_db(self):
        faiss_path = DEFAULT_STORE_PATH + ".faiss"
        if os.path.exists(faiss_path):
            try:
                self._vector_store.load(DEFAULT_STORE_PATH)
                log.info("Auto-loaded vector store: %d vectors", self._vector_store.count())
                self._update_status()
            except Exception as e:
                log.warning("Failed to auto-load vector store: %s", e)

    def _auto_save_db(self):
        if not self._vector_store.is_empty():
            try:
                os.makedirs(os.path.dirname(DEFAULT_STORE_PATH), exist_ok=True)
                self._vector_store.save(DEFAULT_STORE_PATH)
                log.info("Auto-saved vector store: %d vectors", self._vector_store.count())
            except Exception as e:
                log.warning("Failed to auto-save vector store: %s", e)

    def closeEvent(self, event):
        self._auto_save_db()
        for win in self._compare_windows:
            win.close()
        self._case_store.close()
        super().closeEvent(event)
