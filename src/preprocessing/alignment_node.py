from __future__ import annotations

from typing import Any, Dict

from preprocessing.alignment_lib import run_alignment_pipeline
from utils.io_utils import load_config


class AlignmentNode:
    """
    Future Alignment Node -- orchestrates alignment preprocessing.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> None:
        run_alignment_pipeline()

    def close(self) -> None:
        pass


if __name__ == "__main__":
    AlignmentNode().run()
