"""Load orchestration budgets and testcase limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "runtime" / "flow_config.yaml"


@dataclass(frozen=True)
class EgarDrcConfig:
    """Immutable defaults for generation, repair, and review budgets."""

    max_code_generation_compile_iterations: int = 5
    max_code_generation_retry_iterations: int = 20
    max_code_correction_compile_iterations: int = 5
    max_code_correction_iterations: int = 20
    max_testcase_generation_prepare_iterations: int = 20
    max_testcase_correction_prepare_iterations: int = 20
    max_testcase_correction_iterations: int = 20
    max_critic_review_iterations: int = 10
    max_testcases: int = 16

    @classmethod
    def load(cls, config_path: Path = _DEFAULT_CONFIG_PATH) -> EgarDrcConfig:
        """Load flow settings and validate positive limits.

        Args:
            config_path: YAML file containing EGAR-DRC flow settings.

        Returns:
            Validated flow configuration.

        Raises:
            ValueError: If the YAML root or a configured limit is invalid.
            TypeError: If the mapping contains unknown configuration fields.
        """
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"EGAR-DRC config must be a mapping: {config_path}")

        config = cls(**payload)
        for field_name, value in vars(config).items():
            if int(value) < 1:
                raise ValueError(f"{field_name} must be at least 1, got {value}.")
        return config


DEFAULT_EGAR_DRC_CONFIG = EgarDrcConfig.load()
