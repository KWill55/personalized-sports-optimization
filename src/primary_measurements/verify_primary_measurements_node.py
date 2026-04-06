from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/primary_measurements/verify_primary_measurements_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from primary_measurements.verify_primary_measurements_lib import run_verify_primary_measurements_gui
from utils.io_utils import load_config


class VerifyPrimaryMeasurementsNode:
    """Launches GUI for primary measurement verification across cameras."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_verify_primary_measurements_gui(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = VerifyPrimaryMeasurementsNode()
    result = node.run()
    print(f"Primary measurements verification GUI closed: {result}")

