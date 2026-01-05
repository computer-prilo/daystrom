import logging
import os
from logging.config import dictConfig

log = logging.getLogger("daystrom")

if not log.hasHandlers():
    dictConfig(
        {
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "verbose": {
                    "format": "%(asctime)s - %(module)s - %(levelname)s - %(message)s (PID: %(process)d) (%(pathname)s:%(lineno)d)"
                },
            },
            "root": {
                "handlers": ["console"],
                "level": "WARNING",
            },
            "loggers": {
                "daystrom": {
                    "handlers": ["console"],
                    "level": os.getenv("DAYSTROM_LOG_LEVEL", "INFO"),
                    "propagate": False,
                },
            },
        }
    )
