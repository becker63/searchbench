from __future__ import annotations

import logging
import logging.config
import sys
from typing import Literal

import structlog


LogRenderer = Literal["console", "json"]

_DEFAULT_RENDERER: LogRenderer = "console"


def _renderer(renderer: LogRenderer) -> structlog.typing.Processor:
    if renderer == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())


def configure_logging(
    *,
    level: int = logging.INFO,
    renderer: LogRenderer = _DEFAULT_RENDERER,
    force: bool = False,
    cache_logger_on_first_use: bool = True,
) -> None:
    """Configure stdlib logging plus structlog once for the harness.

    This follows structlog's stdlib integration model using
    ``ProcessorFormatter`` plus ``wrap_for_formatter`` so both structlog and
    plain ``logging`` records share the same output path.
    """

    if force:
        structlog.reset_defaults()

    if structlog.is_configured() and not force:
        logging.getLogger().setLevel(level)
        return

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    pre_chain: list[structlog.typing.Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        _renderer(renderer),
                    ],
                    "foreign_pre_chain": pre_chain,
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": level,
                    "propagate": False,
                }
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *pre_chain,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=cache_logger_on_first_use,
    )


__all__ = ["LogRenderer", "configure_logging"]
