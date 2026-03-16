from __future__ import annotations

from typing import Any, Dict

from player_tracking.pose_3d_lib import run_pose_3d_pipeline
from utils.io_utils import load_config


class Pose3dNode:
    """
    Future Pose3d Node -- orchestrates 3D pose process.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> None:
        run_pose_3d_pipeline()

    def close(self) -> None:
        pass


if __name__ == "__main__":
    Pose3dNode().run()
