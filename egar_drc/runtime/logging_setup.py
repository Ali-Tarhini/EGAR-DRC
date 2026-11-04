"""Invocation-local progress logging without root logger changes."""

from __future__ import annotations

import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit
from uuid import uuid4


def sanitize_detail(value: object, api_key: str = "", limit: int = 500) -> str:
    """Filter credentials and request details from a diagnostic line."""
    text = str(value)
    if api_key and api_key != "not-required":
        text = text.replace(api_key, "[REDACTED]")
    text = text.splitlines()[0] if text.splitlines() else ""
    text = re.split(
        r"(?i)\b(?:prompt|messages|request[ _-]?body|response[ _-]?body|body)[\"']?\s*[:=]", text,
        maxsplit=1,
    )[0]

    def public_origin(match: re.Match[str]) -> str:
        """Keep only a URL's public origin."""
        try:
            parsed = urlsplit(match.group())
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{parsed.scheme}://{parsed.hostname or ''}{port}"
        except ValueError:
            return "[REDACTED URL]"

    text = re.sub(r"https?://[^\s\"'<>]+", public_origin, text)
    text = re.sub(
        r"(?i)\b(?:authorization|api[_-]?key|access[_-]?token|password)\s*[:=]\s*[^,;]+",
        "credential=[REDACTED]", text,
    )
    return text[:limit]


@contextmanager
def run_logger(output_dir: Path, level: str = "INFO") -> Iterator[logging.Logger]:
    """Manage invocation-local stderr and append-only file logging."""
    logger = logging.Logger(f"egar_drc.runner.{uuid4().hex}", level=level.upper())
    logger.propagate = False
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []
    try:
        handlers.append(logging.StreamHandler(sys.stderr))
        handlers.append(logging.FileHandler(output_dir / "run.log", encoding="utf-8"))
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()