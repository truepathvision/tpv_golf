from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QLineEdit, QTextEdit, QHeaderView,
    QAbstractItemView, QMessageBox, QTabWidget, QWidget,
)
from PySide6.QtCore import Qt, Signal

from core.case_store import CaseStore, Case


class NewCaseDialog(QDialog):
    """Small dialog to create a new case."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Case")
        self.setMinimumWidth(400)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Case name")

        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText("Description (optional)")
        self._desc_edit.setMaximumHeight(80)

        self._ok_btn = QPushButton("Create")
        self._ok_btn.clicked.connect(self.accept)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._ok_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Case Name:"))
        layout.addWidget(self._name_edit)
        layout.addWidget(QLabel("Description:"))
        layout.addWidget(self._desc_edit)
        layout.addLayout(btn_row)

    @property
    def case_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def case_description(self) -> str:
        return self._desc_edit.toPlainText().strip()


class CaseManagerDialog(QDialog):
    """Full case management dialog with tabs for cases list and audit log."""

    case_opened = Signal(int)  # case_id
    case_created = Signal(int)  # case_id

    def __init__(self, case_store: CaseStore, active_case_id: int | None = None, parent=None):
        super().__init__(parent)
        self._store = case_store
        self._active_case_id = active_case_id

        self.setWindowTitle("Case Manager")
        self.resize(700, 500)

        self._tabs = QTabWidget()

        # --- Cases tab ---
        cases_tab = QWidget()
        self._cases_table = QTableWidget()
        self._cases_table.setColumnCount(5)
        self._cases_table.setHorizontalHeaderLabels(["ID", "Name", "Status", "Images", "Updated"])
        self._cases_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._cases_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._cases_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._cases_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._cases_table.doubleClicked.connect(self._on_open)

        self._new_btn = QPushButton("New Case")
        self._new_btn.clicked.connect(self._on_new)
        self._open_btn = QPushButton("Open Case")
        self._open_btn.clicked.connect(self._on_open)
        self._close_btn = QPushButton("Close Case")
        self._close_btn.clicked.connect(self._on_close_case)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addWidget(self._close_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._delete_btn)

        cases_layout = QVBoxLayout(cases_tab)
        cases_layout.addWidget(self._cases_table, 1)
        cases_layout.addLayout(btn_row)

        # --- Audit log tab ---
        audit_tab = QWidget()
        self._audit_table = QTableWidget()
        self._audit_table.setColumnCount(4)
        self._audit_table.setHorizontalHeaderLabels(["Timestamp", "Action", "Case", "Details"])
        self._audit_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._audit_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        audit_layout = QVBoxLayout(audit_tab)
        audit_layout.addWidget(self._audit_table, 1)

        # --- Notes tab ---
        notes_tab = QWidget()
        self._notes_table = QTableWidget()
        self._notes_table.setColumnCount(3)
        self._notes_table.setHorizontalHeaderLabels(["Timestamp", "Image", "Note"])
        self._notes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._notes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        notes_layout = QVBoxLayout(notes_tab)
        notes_layout.addWidget(self._notes_table, 1)

        self._tabs.addTab(cases_tab, "Cases")
        self._tabs.addTab(audit_tab, "Audit Log")
        self._tabs.addTab(notes_tab, "Notes")

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._tabs)

        self._refresh()

    def _selected_case_id(self) -> int | None:
        rows = self._cases_table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return int(self._cases_table.item(row, 0).text())

    def _refresh(self):
        self._refresh_cases()
        self._refresh_audit()
        self._refresh_notes()

    def _refresh_cases(self):
        cases = self._store.list_cases()
        self._cases_table.setRowCount(len(cases))
        for i, c in enumerate(cases):
            self._cases_table.setItem(i, 0, QTableWidgetItem(str(c.id)))
            self._cases_table.setItem(i, 1, QTableWidgetItem(c.name))
            self._cases_table.setItem(i, 2, QTableWidgetItem(c.status))
            self._cases_table.setItem(i, 3, QTableWidgetItem(str(self._store.image_count(c.id))))
            self._cases_table.setItem(i, 4, QTableWidgetItem(c.updated_at))

    def _refresh_audit(self):
        if self._active_case_id is None:
            self._audit_table.setRowCount(0)
            return
        entries = self._store.get_audit_log(self._active_case_id)
        self._audit_table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            self._audit_table.setItem(i, 0, QTableWidgetItem(e.timestamp))
            self._audit_table.setItem(i, 1, QTableWidgetItem(e.action))
            self._audit_table.setItem(i, 2, QTableWidgetItem(str(e.case_id or "")))
            details_str = ", ".join(f"{k}={v}" for k, v in e.details.items()) if e.details else ""
            self._audit_table.setItem(i, 3, QTableWidgetItem(details_str))

    def _refresh_notes(self):
        if self._active_case_id is None:
            self._notes_table.setRowCount(0)
            return
        notes = self._store.get_notes(self._active_case_id)
        self._notes_table.setRowCount(len(notes))
        for i, n in enumerate(notes):
            self._notes_table.setItem(i, 0, QTableWidgetItem(n.get("created_at", "")))
            self._notes_table.setItem(i, 1, QTableWidgetItem(str(n.get("image_id", ""))))
            self._notes_table.setItem(i, 2, QTableWidgetItem(n.get("note_text", "")))

    def _on_new(self):
        dlg = NewCaseDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.case_name:
            case = self._store.create_case(dlg.case_name, dlg.case_description)
            self._store.log_action(case.id, "create_case", {"name": case.name})
            self.case_created.emit(case.id)
            self._active_case_id = case.id
            self._refresh()

    def _on_open(self):
        case_id = self._selected_case_id()
        if case_id is not None:
            self._active_case_id = case_id
            self.case_opened.emit(case_id)
            self._refresh()
            self.accept()

    def _on_close_case(self):
        case_id = self._selected_case_id()
        if case_id is not None:
            self._store.close_case(case_id)
            self._refresh()

    def _on_delete(self):
        case_id = self._selected_case_id()
        if case_id is None:
            return
        reply = QMessageBox.question(
            self, "Delete Case",
            f"Delete case #{case_id}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete_case(case_id)
            if self._active_case_id == case_id:
                self._active_case_id = None
            self._refresh()
