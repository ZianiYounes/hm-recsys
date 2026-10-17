"""Suivi MLflow. Ne bloque jamais le pipeline si MLflow pose probleme."""
from __future__ import annotations

import os
from typing import Any

from loguru import logger


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out |= flatten(v, key)
        else:
            out[key] = v
    return out


def _mlflow(experiment: str):
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    return mlflow


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str] | None = None,
) -> None:
    """Un run complet en un appel (params + metriques finales)."""
    with MlflowRun(experiment, run_name, params) as run:
        run.log(metrics)
        for a in artifacts or []:
            run.artifact(a)


class MlflowRun:
    """Run MLflow pour un entrainement : metriques par epoch, artefacts, tolerant aux pannes."""

    def __init__(self, experiment: str, run_name: str, params: dict[str, Any]):
        self._ml = None
        try:
            ml = _mlflow(experiment)
            ml.start_run(run_name=run_name)
            ml.log_params({k: str(v) for k, v in flatten(params).items()})
            self._ml = ml
        except Exception as e:  # le tracking est un bonus, pas un point de blocage
            logger.warning(f"MLflow indisponible ({type(e).__name__}: {e}), run non loggue")

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._ml is not None:
            try:
                self._ml.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metrics : {e}")

    def artifact(self, path: str) -> None:
        if self._ml is not None:
            try:
                self._ml.log_artifact(path)
            except Exception as e:
                logger.warning(f"MLflow log_artifact : {e}")

    def end(self) -> None:
        if self._ml is not None:
            self._ml.end_run()
            self._ml = None

    def __enter__(self) -> MlflowRun:
        return self

    def __exit__(self, *exc) -> None:
        self.end()
