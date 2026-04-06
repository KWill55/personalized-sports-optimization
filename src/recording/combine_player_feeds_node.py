from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/recording/combine_player_feeds_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from recording.combine_player_feeds_lib import run_combine_player_feeds_pipeline
from utils.io_utils import load_config


class CombinePlayerFeedsNode:
    """Orchestrates left/right player-feed combining."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_combine_player_feeds_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = CombinePlayerFeedsNode()
    result = node.run()
    print(f"Combine player feeds pipeline complete: {result}")
