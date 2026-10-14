"""SQLite-backed time-series store for system and per-process metrics.

Every module that needs history (memory forecasting, workload classification,
log correlation) reads from here, so the collector and the analytics are
decoupled.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS system_metrics (
    ts REAL NOT NULL,
    cpu_percent REAL,
    mem_percent REAL,
    mem_used_mb REAL,
    mem_available_mb REAL,
    swap_percent REAL,
    load1 REAL,
    load5 REAL,
    load15 REAL,
    ctx_switches INTEGER,
    interrupts INTEGER,
    process_count INTEGER,
    thread_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_system_ts ON system_metrics(ts);

CREATE TABLE IF NOT EXISTS process_metrics (
    ts REAL NOT NULL,
    pid INTEGER NOT NULL,
    name TEXT,
    state TEXT,
    cpu_percent REAL,
    rss_mb REAL,
    num_threads INTEGER,
    vol_ctx INTEGER,
    invol_ctx INTEGER,
    read_bytes INTEGER,
    write_bytes INTEGER,
    nice INTEGER
);
CREATE INDEX IF NOT EXISTS idx_proc_ts ON process_metrics(ts);
CREATE INDEX IF NOT EXISTS idx_proc_pid ON process_metrics(pid, ts);
"""

SYSTEM_COLUMNS = [
    "ts", "cpu_percent", "mem_percent", "mem_used_mb", "mem_available_mb", "swap_percent",
    "load1", "load5", "load15", "ctx_switches", "interrupts", "process_count", "thread_count",
]
PROCESS_COLUMNS = [
    "ts", "pid", "name", "state", "cpu_percent", "rss_mb", "num_threads",
    "vol_ctx", "invol_ctx", "read_bytes", "write_bytes", "nice",
]


class MetricStore:
    """Thread-safe wrapper around a SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.db_path
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    # ---------- writes ----------
    def insert_system(self, row: dict) -> None:
        values = [row.get(c) for c in SYSTEM_COLUMNS]
        placeholders = ",".join("?" * len(SYSTEM_COLUMNS))
        with self._cursor() as cur:
            cur.execute(f"INSERT INTO system_metrics ({','.join(SYSTEM_COLUMNS)}) VALUES ({placeholders})", values)

    def insert_processes(self, rows: Iterable[dict]) -> None:
        values = [[r.get(c) for c in PROCESS_COLUMNS] for r in rows]
        if not values:
            return
        placeholders = ",".join("?" * len(PROCESS_COLUMNS))
        with self._cursor() as cur:
            cur.executemany(
                f"INSERT INTO process_metrics ({','.join(PROCESS_COLUMNS)}) VALUES ({placeholders})", values
            )

    def prune(self, retention_hours: float | None = None) -> None:
        cutoff = time.time() - 3600 * (retention_hours or settings.retention_hours)
        with self._cursor() as cur:
            cur.execute("DELETE FROM system_metrics WHERE ts < ?", (cutoff,))
            cur.execute("DELETE FROM process_metrics WHERE ts < ?", (cutoff,))

    # ---------- reads ----------
    def system_history(self, minutes: float = 60) -> pd.DataFrame:
        cutoff = time.time() - minutes * 60
        with self._lock:
            df = pd.read_sql_query(
                "SELECT * FROM system_metrics WHERE ts >= ? ORDER BY ts", self._conn, params=(cutoff,)
            )
        if not df.empty:
            df["time"] = pd.to_datetime(df["ts"], unit="s")
        return df

    def process_history(self, pid: int | None = None, minutes: float = 60) -> pd.DataFrame:
        cutoff = time.time() - minutes * 60
        query = "SELECT * FROM process_metrics WHERE ts >= ?"
        params: list = [cutoff]
        if pid is not None:
            query += " AND pid = ?"
            params.append(pid)
        query += " ORDER BY ts"
        with self._lock:
            df = pd.read_sql_query(query, self._conn, params=params)
        if not df.empty:
            df["time"] = pd.to_datetime(df["ts"], unit="s")
        return df

    def row_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
