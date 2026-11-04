"""Logging configuration for the MCP server."""

from __future__ import annotations

import logging
import os


def configure_logging() -> logging.Logger:
    """Configure a conservative stderr logger for stdio MCP transport."""
    level_name = os.getenv("EB_CR_AFM_MCP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger("eb_cr_afm_mcp")

