"""Optional Hugging Face adapter for preparing the external benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from egar_drc.dataset_preparation import materialize_dataset


def main(argv: list[str] | None = None) -> None:
    """Load both test splits at one immutable revision and publish validated files.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare the external Rule2DRC benchmark repository")
    parser.add_argument("--output-dir", required=True, type=Path, help="New destination directory (must not exist)")
    parser.add_argument("--revision", default="main", help="Dataset branch, tag or commit (default: main)")
    args = parser.parse_args(argv)
    repository = "jusjinuk/Rule2DRC"
    # Materialize the Rule2DRC benchmark hosted on Hugging Face.
    revision = HfApi().dataset_info(repo_id=repository, revision=args.revision).sha
    if not revision:
        raise ValueError("Hugging Face did not return a snapshot commit SHA")
    tasks = load_dataset(repository, "tasks", split="test", revision=revision)
    testcases = load_dataset(repository, "testcases", split="test", revision=revision)
    destination = materialize_dataset(
        tasks,
        testcases,
        args.output_dir,
        repository=repository,
        requested_revision=args.revision,
        resolved_revision=revision,
        fingerprints={"tasks": getattr(tasks, "_fingerprint", None), "testcases": getattr(testcases, "_fingerprint", None)},
    )
    print(f"Prepared benchmark repository: {destination}")


if __name__ == "__main__":
    main()