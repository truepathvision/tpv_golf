import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".tpv_golf")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "cases.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS case_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'source',
    embedding_blob BLOB,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    details_json TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES case_images(id) ON DELETE SET NULL,
    note_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass
class Case:
    id: int
    name: str
    description: str
    created_at: str
    updated_at: str
    status: str


@dataclass
class AuditEntry:
    id: int
    case_id: int | None
    action: str
    details: dict
    timestamp: str


class CaseStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    # ---- Case CRUD ----

    def create_case(self, name: str, description: str = "") -> Case:
        now = self._now()
        cur = self._conn.execute(
            "INSERT INTO cases (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, description, now, now),
        )
        self._conn.commit()
        return self.get_case(cur.lastrowid)

    def get_case(self, case_id: int) -> Case:
        row = self._conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not row:
            raise ValueError(f"Case {case_id} not found")
        return Case(**dict(row))

    def list_cases(self, status: str | None = None) -> list[Case]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY updated_at DESC", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM cases ORDER BY updated_at DESC"
            ).fetchall()
        return [Case(**dict(r)) for r in rows]

    def update_case(self, case_id: int, **kwargs):
        allowed = {"name", "description", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [case_id]
        self._conn.execute(f"UPDATE cases SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    def close_case(self, case_id: int):
        self.update_case(case_id, status="closed")
        self.log_action(case_id, "close_case")

    def delete_case(self, case_id: int):
        self._conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        self._conn.commit()

    # ---- Case Images ----

    def add_image(self, case_id: int, image_path: str, role: str = "source",
                  embedding: bytes | None = None) -> int:
        now = self._now()
        cur = self._conn.execute(
            "INSERT INTO case_images (case_id, image_path, role, embedding_blob, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, image_path, role, embedding, now),
        )
        self._conn.commit()
        self.update_case(case_id)
        return cur.lastrowid

    def get_images(self, case_id: int, role: str | None = None) -> list[dict]:
        if role:
            rows = self._conn.execute(
                "SELECT * FROM case_images WHERE case_id = ? AND role = ? ORDER BY added_at",
                (case_id, role),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM case_images WHERE case_id = ? ORDER BY added_at",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Audit Log ----

    def log_action(self, case_id: int | None, action: str, details: dict | None = None):
        self._conn.execute(
            "INSERT INTO audit_log (case_id, action, details_json, timestamp) VALUES (?, ?, ?, ?)",
            (case_id, action, json.dumps(details or {}), self._now()),
        )
        self._conn.commit()

    def get_audit_log(self, case_id: int) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE case_id = ? ORDER BY timestamp DESC",
            (case_id,),
        ).fetchall()
        return [
            AuditEntry(
                id=r["id"],
                case_id=r["case_id"],
                action=r["action"],
                details=json.loads(r["details_json"]),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ---- Notes ----

    def add_note(self, case_id: int, note_text: str, image_id: int | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO case_notes (case_id, image_id, note_text, created_at) VALUES (?, ?, ?, ?)",
            (case_id, image_id, note_text, self._now()),
        )
        self._conn.commit()
        self.log_action(case_id, "add_note", {"note_text": note_text[:200]})
        return cur.lastrowid

    def get_notes(self, case_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM case_notes WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Stats ----

    def image_count(self, case_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM case_images WHERE case_id = ?", (case_id,)
        ).fetchone()
        return row["cnt"]

    def close(self):
        self._conn.close()
