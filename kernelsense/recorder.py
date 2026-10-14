"""Background metric recorder.

Samples system and per-process metrics every ``interval`` seconds and writes
them to the MetricStore. Used by the dashboard (embedded thread) and by the
``kernelsense record`` CLI command (foreground).
"""

from __future__ import annotations

import threading
import time

from .config import settings
from .monitoring import PsutilCollector, processes_to_rows
from .storage import MetricStore


class Recorder:
    def __init__(self, store: MetricStore | None = None, interval: float | None = None, top_n: int | None = None):
        self.store = store or MetricStore()
        self.interval = interval or settings.sample_interval_s
        self.top_n = top_n or settings.top_n_processes
        self.collector = PsutilCollector()
        self.latest_system = None
        self.latest_processes: list = []
        self.samples = 0
        self.overhead_ms = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def sample_once(self) -> None:
        t0 = time.perf_counter()
        sysm = self.collector.system()
        procs = self.collector.processes()
        # Persist the busiest processes (by CPU, then memory) to bound database growth.
        top = sorted(procs, key=lambda p: (p.cpu_percent, p.rss_mb), reverse=True)[: self.top_n]
        self.store.insert_system(sysm.to_row())
        self.store.insert_processes(processes_to_rows(top, sysm.ts))
        with self._lock:
            self.latest_system = sysm
            self.latest_processes = procs
            self.samples += 1
            self.overhead_ms = (time.perf_counter() - t0) * 1000
        if self.samples % 300 == 0:
            self.store.prune()

    def snapshot(self):
        with self._lock:
            return self.latest_system, list(self.latest_processes)

    def _loop(self) -> None:
        while not self._stop.is_set():
            start = time.time()
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 - a failed sample must not kill the recorder
                pass
            self._stop.wait(max(0.0, self.interval - (time.time() - start)))

    def start(self) -> "Recorder":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kernelsense-recorder", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
