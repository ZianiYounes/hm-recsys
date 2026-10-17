from __future__ import annotations

import time
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    logger.info(f"{name} : {time.perf_counter() - t0:.1f}s")
