"""Central configuration. Values can be overridden through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("KS_DATA_DIR", PROJECT_ROOT / "data")))
    sample_interval_s: float = field(default_factory=lambda: _env_float("KS_SAMPLE_INTERVAL", 2.0))
    retention_hours: float = field(default_factory=lambda: _env_float("KS_RETENTION_HOURS", 24.0))
    top_n_processes: int = 30
    memory_alert_pct: float = field(default_factory=lambda: _env_float("KS_MEM_ALERT_PCT", 90.0))
    collector: str = field(default_factory=lambda: os.environ.get("KS_COLLECTOR", "auto"))
    # Resource actions are simulated unless explicitly enabled.
    dry_run: bool = field(default_factory=lambda: os.environ.get("KS_DRY_RUN", "1") != "0")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kernelsense.db"

    @property
    def action_log_path(self) -> Path:
        return self.data_dir / "actions.log"


settings = Settings()
