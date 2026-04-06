from __future__ import annotations

from typing import Any, Dict

from evaluation.golden_template_lib import run_golden_template_pipeline
from utils.io_utils import load_config


class GoldenTemplateNode:
    """Orchestrates golden-template comparison/evaluation."""

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(
        self,
        *,
        compare_athlete: str | None = None,
        compare_session: str | None = None,
    ) -> dict[str, Any]:
        run_cfg = dict(self.cfg)
        if compare_athlete:
            run_cfg["athlete"] = compare_athlete
        if compare_session:
            run_cfg["session"] = compare_session
        return run_golden_template_pipeline(run_cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    result = GoldenTemplateNode().run()
    print(f"Golden template evaluation complete: {result}")
