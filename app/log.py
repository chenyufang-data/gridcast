"""Logging conventions (see CONTRIBUTING.md): one root handler, one logger per module.

In any module::

    log = logging.getLogger(__name__)

Entry points call :func:`configure_logging` once: ``app.main`` at import, CLI scripts
in ``main()``. ``LOG_LEVEL`` picks the level (default ``INFO``). Library code never
``print()``; ruff rule T20 enforces it.
"""

from __future__ import annotations

import logging
import os
import sys
import time

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
_MARK = "_gridcast_handler"


def configure_logging(level: str | None = None) -> None:
    """Idempotent root setup: UTC timestamps on stderr; level from arg, LOG_LEVEL, INFO.

    Adds our handler once and leaves any others alone (pytest's caplog, uvicorn), so
    calling it from several entry points or tests is safe.
    """
    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    if not any(getattr(h, _MARK, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        setattr(handler, _MARK, True)
        root.addHandler(handler)
    root.setLevel(resolved)
