from __future__ import annotations

from typing import Any, Dict

from recording.recording_capture_gui import RecordingCaptureGui
from utils.io_utils import load_config


class RecordingNode:
    """
    Thin orchestrator for recording subsystem.

    Main entrypoint used by src/main.py.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run_capture_gui(self) -> None:
        gui = RecordingCaptureGui(cfg=self.cfg)
        gui.run()

    def run(self) -> None:
        self.run_capture_gui()


if __name__ == "__main__":
    RecordingNode().run()
