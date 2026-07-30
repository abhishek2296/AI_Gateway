"""
Logging configuration for the AI Gateway.

Per the "Structured logging" engineering principle, every module should log
via ``logging.getLogger(__name__)`` rather than printing or configuring
logging itself. This module provides the single place that configures the
root logging setup (format, level, handlers) for the whole process; it is
invoked once, early in ``core/lifespan.py``'s startup sequence, so every
logger created afterwards (including ones instantiated at import time in
other modules) shares the same formatting and destination.
"""

import logging


def setup_logging() -> logging.Logger:
    """
    Configure process-wide logging and return the gateway's named logger.

    Calls ``logging.basicConfig`` to install a single stream handler on the
    root logger with a consistent, human-readable format
    (``timestamp | LEVEL | logger.name | message``). Because
    ``logging.basicConfig`` is a no-op if the root logger already has
    handlers configured, this function is safe to treat as "configure once
    at startup" — calling it again later would not change the existing
    configuration.

    The level is currently fixed at ``logging.INFO`` rather than read from
    ``Settings.LOG_LEVEL``; call sites needing a different level should
    configure it directly on the returned/relevant logger until this is
    wired up to read from settings.

    Returns:
        The ``"ai_gateway"`` named logger. Most modules should still prefer
        their own ``logging.getLogger(__name__)`` logger (which propagates
        up to the root logger configured here) rather than using this
        returned logger directly, so log records are attributed to the
        module that emitted them.

    Example:
        >>> logger = setup_logging()
        >>> logger.name
        'ai_gateway'
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    return logging.getLogger("ai_gateway")