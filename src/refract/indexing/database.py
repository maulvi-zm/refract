from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from refract.core.models import MethodInfo, RepositoryIndex, SmellLocation, SmellType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS methods (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    class_name  TEXT    NOT NULL,
    file        TEXT    NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    cyclomatic_complexity INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS method_calls (
    method_id   INTEGER NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    callee      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS method_parameters (
    method_id   INTEGER NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    parameter   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS smells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    smell       TEXT    NOT NULL,
    file        TEXT    NOT NULL,
    line        INTEGER NOT NULL,
    identifier  TEXT    NOT NULL,
    detail      TEXT    NOT NULL
);
"""


def save(index: RepositoryIndex, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)

    try:
        with conn:
            for table in ("method_parameters", "method_calls", "methods", "smells"):
                conn.execute(f"DELETE FROM {table}")

            for method in index.methods:
                _insert_method(conn, method)

            conn.executemany(
                "INSERT INTO smells (smell, file, line, identifier, detail) VALUES (?, ?, ?, ?, ?)",
                [
                    (s.smell.value, str(s.file), s.line, s.identifier, s.detail)
                    for s in index.smells
                ],
            )
    finally:
        conn.close()


def _insert_method(conn: sqlite3.Connection, method: MethodInfo) -> None:
    cursor = conn.execute(
        "INSERT INTO methods "
        "(name, class_name, file, start_line, end_line, cyclomatic_complexity) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            method.name,
            method.class_name,
            str(method.file),
            method.start_line,
            method.end_line,
            method.cyclomatic_complexity,
        ),
    )
    method_id = cursor.lastrowid

    conn.executemany(
        "INSERT INTO method_calls (method_id, callee) VALUES (?, ?)",
        [(method_id, c) for c in method.calls],
    )
    conn.executemany(
        "INSERT INTO method_parameters (method_id, parameter) VALUES (?, ?)",
        [(method_id, p) for p in method.parameters],
    )


def load(db_path: Path) -> RepositoryIndex:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        calls = _grouped(conn, "method_calls", "callee")
        parameters = _grouped(conn, "method_parameters", "parameter")

        # insertion order matters for name resolution
        methods = [
            MethodInfo(
                name=row["name"],
                class_name=row["class_name"],
                file=Path(row["file"]),
                start_line=row["start_line"],
                end_line=row["end_line"],
                cyclomatic_complexity=row["cyclomatic_complexity"],
                calls=calls.get(row["id"], []),
                parameters=parameters.get(row["id"], []),
            )
            for row in conn.execute("SELECT * FROM methods ORDER BY id")
        ]
        smells = [
            SmellLocation(
                smell=SmellType(row["smell"]),
                file=Path(row["file"]),
                line=row["line"],
                identifier=row["identifier"],
                detail=row["detail"],
            )
            for row in conn.execute("SELECT * FROM smells ORDER BY id")
        ]
    finally:
        conn.close()

    return RepositoryIndex(methods=methods, smells=smells)


def _grouped(conn: sqlite3.Connection, table: str, column: str) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for row in conn.execute(f"SELECT method_id, {column} FROM {table}"):
        grouped[row["method_id"]].append(row[column])
    return grouped
