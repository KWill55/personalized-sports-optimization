from __future__ import annotations

from typing import Any, Dict

from preprocessing.phases_lib import run_phases_pipeline
from utils.io_utils import load_config


class PhasesNode:
    """
    Future Phases Node -- orchestrates phase preprocessing.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> None:
        run_phases_pipeline()

    def close(self) -> None:
        pass


if __name__ == "__main__":
    PhasesNode().run()
