"""Provider configuration with environment-resolved credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Validate provider settings and request limits."""

    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: str | None = None
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    retry_attempts: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=16, ge=1)
    additional_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_strings(self) -> ModelConfig:
        """Reject blank model and endpoint values without rewriting their text."""
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort must not be blank")
        return self

    @property
    def api_key(self) -> str:
        """Resolve the API key from the configured environment variable."""
        return os.getenv(self.api_key_env, "not-required")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate provider settings from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Model config must be a mapping: {path}")
        return cls.model_validate(payload)
