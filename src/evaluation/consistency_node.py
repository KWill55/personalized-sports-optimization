from __future__ import annotations

from typing import Any, Dict

from evaluation.consistency_lib import run_consistency_pipeline
from utils.io_utils import load_config


class ConsistencyNode:
    """Orchestrates athlete form-consistency viewer."""

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_consistency_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    result = ConsistencyNode().run()
    print(f"Consistency evaluation complete: {result}")
