"""Prometheus exporter (text exposition format, no extra dependencies).

    python -m kernelsense export --port 9877
    curl localhost:9877/metrics

Scrape it from Prometheus and import ``grafana/kernelsense-dashboard.json``.
Each scrape takes a fresh sample from the /proc collector (rates are computed
against the previous scrape).
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .monitoring import detect_anomalies, make_collector
from .monitoring import procfs


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class MetricsBuilder:
    """Collects samples and renders each metric family as one contiguous group (required by the format)."""

    def __init__(self) -> None:
        self.families: dict[str, tuple[str, str, list[str]]] = {}

    def add(self, name: str, value, help_: str, kind: str = "gauge", labels: dict | None = None) -> None:
        fam = self.families.setdefault(name, (help_, kind, []))
        lab = "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items()) + "}" if labels else ""
        fam[2].append(f"{name}{lab} {float(value):.6g}")

    def text(self) -> str:
        out = []
        for name, (help_, kind, samples) in self.families.items():
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}", *samples]
        return "\n".join(out) + "\n"


class KernelSenseExporter:
    def __init__(self, top_n: int = 15) -> None:
        self.collector = make_collector()
        self.collector.processes()
        self.top_n = top_n
        self._lock = threading.Lock()
        self.scrapes = 0

    def render(self) -> str:
        with self._lock:
            t0 = time.perf_counter()
            s = self.collector.system()
            procs = self.collector.processes()
            m = MetricsBuilder()
            m.add("kernelsense_cpu_utilization_percent", s.cpu_percent, "Total CPU utilisation")
            for i, v in enumerate(s.per_cpu):
                m.add("kernelsense_cpu_core_utilization_percent", v, "Per-core CPU utilisation", labels={"core": i})
            m.add("kernelsense_memory_used_percent", s.mem_percent, "Memory in use (MemTotal - MemAvailable)")
            m.add("kernelsense_memory_available_bytes", s.mem_available_mb * 2**20, "MemAvailable")
            m.add("kernelsense_swap_used_percent", s.swap_percent, "Swap in use")
            for period, v in (("1m", s.load1), ("5m", s.load5), ("15m", s.load15)):
                m.add("kernelsense_load_average", v, "Load average", labels={"period": period})
            m.add("kernelsense_processes", s.process_count, "Number of processes")
            m.add("kernelsense_runnable_tasks", s.runnable, "Runnable tasks (from /proc/loadavg)")
            m.add("kernelsense_context_switches_total", s.ctx_switches, "Context switches since boot", "counter")
            for res, val in (s.pressure or {}).items():
                for kind, windows in (val or {}).items():
                    for w in ("avg10", "avg60", "avg300"):
                        if w in windows:
                            m.add("kernelsense_pressure_stall_percent", windows[w], "Pressure Stall Information",
                                  labels={"resource": res, "kind": kind, "window": w})
            for k, v in procfs.read_vmstat(procfs.VMSTAT_KEYS).items():
                m.add(f"kernelsense_vmstat_{k}_total", v, f"/proc/vmstat {k}", "counter")
            states: dict[str, int] = {}
            classes: dict[str, int] = {}
            for p in procs:
                states[p.state] = states.get(p.state, 0) + 1
                classes[p.workload_class] = classes.get(p.workload_class, 0) + 1
            for st, n in states.items():
                m.add("kernelsense_processes_by_state", n, "Processes per scheduler state", labels={"state": st})
            for c, n in classes.items():
                m.add("kernelsense_processes_by_class", n, "Processes per workload class", labels={"class": c})
            for p in sorted(procs, key=lambda x: (x.cpu_percent, x.rss_mb), reverse=True)[: self.top_n]:
                lab = {"pid": p.pid, "name": p.name, "class": p.workload_class}
                m.add("kernelsense_process_cpu_percent", p.cpu_percent, "Process CPU %", labels=lab)
                m.add("kernelsense_process_resident_bytes", p.rss_mb * 2**20, "Process RSS", labels=lab)
                m.add("kernelsense_process_voluntary_switches_per_second", p.vol_ctx_rate, "Voluntary switches/s", labels=lab)
                m.add("kernelsense_process_involuntary_switches_per_second", p.invol_ctx_rate, "Involuntary switches/s", labels=lab)
            kinds: dict[str, int] = {}
            for a in detect_anomalies(procs):
                kinds[a.kind] = kinds.get(a.kind, 0) + 1
            for k, n in kinds.items():
                m.add("kernelsense_anomalies", n, "Process anomalies detected", labels={"kind": k})
            self.scrapes += 1
            m.add("kernelsense_scrapes_total", self.scrapes, "Scrapes served", "counter")
            m.add("kernelsense_scrape_duration_seconds", time.perf_counter() - t0, "Time to collect this scrape")
            return m.text()


def serve(port: int = 9877, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    exporter = KernelSenseExporter()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            if self.path.rstrip("/") not in ("/metrics", ""):
                self.send_response(404)
                self.end_headers()
                return
            body = exporter.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence default request logging
            pass

    return ThreadingHTTPServer((host, port), Handler)
