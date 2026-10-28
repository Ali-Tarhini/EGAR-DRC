"""Validated prompt descriptor loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class AgentDescriptor(BaseModel):
    """Static prompt and model settings for one agent role."""

    agent_name: str
    system_prompt: str
    user_prompt_template: str
    prompt_input_fields: list[str] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    additional_model_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_template(self) -> AgentDescriptor:
        """Require named placeholders to be declared, allowing unused inputs."""
        placeholders: set[str] = set()
        for _, field_name, _, _ in Formatter().parse(self.user_prompt_template):
            if field_name is None:
                continue
            if field_name.isdigit():
                raise ValueError("Positional prompt placeholders are not supported")
            placeholders.add(field_name)
        undeclared = placeholders - set(self.prompt_input_fields)
        if undeclared:
            raise ValueError(f"Undeclared prompt fields: {sorted(undeclared)}")
        return self

    def render_user_prompt(self, values: Mapping[str, Any]) -> str:
        """Render the prompt after checking all declared inputs."""
        missing = sorted(field for field in self.prompt_input_fields if field not in values)
        if missing:
            raise ValueError(f"Missing prompt fields: {missing}")
        return self.user_prompt_template.format(**values)


def load_descriptor(path: str | Path) -> AgentDescriptor:
    """Load one descriptor from YAML."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Descriptor must be a mapping: {path}")
    return AgentDescriptor.model_validate(payload)
