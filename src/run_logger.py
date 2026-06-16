"""Run log: sequential writes to logs/evaluation-N.txt (N increments each run)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, logs_dir: Path | str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self._next_run_id()
        self.log_path = self.logs_dir / f"evaluation-{self.run_id}.txt"
        self._file = open(self.log_path, "a", encoding="utf-8")
        self.section(f"Run #{self.run_id} — {datetime.now().isoformat(timespec='seconds')}")

    @staticmethod
    def _next_run_id() -> int:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return 1
        max_id = 0
        for path in logs_dir.glob("evaluation-*.txt"):
            match = re.fullmatch(r"evaluation-(\d+)\.txt", path.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id + 1

    def section(self, title: str) -> None:
        bar = "=" * 60
        self.log(f"\n{bar}\n{title}\n{bar}")

    def log(self, message: str = "") -> None:
        print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
