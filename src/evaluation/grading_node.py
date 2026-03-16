from __future__ import annotations

from typing import Any, Dict

from utils.io_utils import load_config

class GradingNode:
    """
    Future Grading Node -- orchestrates grading process
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> None:
        print("Running grading node")


if __name__ == "__main__":
    GradingNode().run()
