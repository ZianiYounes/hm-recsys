from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: Path | None = None, level: str = "INFO"):
    """Console lisible + fichier de log optionnel."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="DEBUG", rotation="10 MB", encoding="utf-8")
    return logger
