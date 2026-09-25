import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    """Lightweight experiment tracking class for logging hyperparams, metrics, and metadata."""

    def __init__(self, experiment_id: str, experiments_dir: Union[str, Path] = "experiments"):
        self.experiment_id = experiment_id
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.experiments_dir / f"{experiment_id}_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}
        self.params: Dict[str, Any] = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters or runtime configuration."""
        self.params.update(params)
        self._save_run_summary()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Logs evaluation metrics (e.g. F0.5, precision, recall)."""
        self.metrics.update(metrics)
        self._save_run_summary()

    def _save_run_summary(self) -> None:
        summary_path = self.run_dir / "summary.json"
        summary_data = {
            "experiment_id": self.experiment_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": self.params,
            "metrics": self.metrics,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    def get_run_dir(self) -> Path:
        return self.run_dir
