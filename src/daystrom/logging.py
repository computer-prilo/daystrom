import logging
import os
from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "handlers": {
            "console": {
                "class": logging.StreamHandler,
            }
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
