from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict


@dataclass
class CsvLogger:
    log_dir: Path

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.csv"
        self.start_time = time.time()
        if not self.metrics_path.exists():
            self.metrics_path.write_text("step,wall_time_s,metric,value\n", encoding="utf-8")

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        wall = time.time() - self.start_time
        lines = [
            f"{step},{wall:.3f},{name},{value}" for name, value in metrics.items()
        ]
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def log_text(self, name: str, text: str) -> None:
        path = self.log_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")

    def log_config(self, config: Dict[str, Any]) -> None:
        path = self.log_dir / "config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
