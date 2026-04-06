"""Consistency viewer built on top of the golden-template viewer implementation."""

from __future__ import annotations

from typing import Any

from evaluation.golden_template_lib import run_golden_template_viewer


def run_consistency_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    athlete = str(cfg.get("athlete", "") or "")
    session = str(cfg.get("session", "") or "")
    if not athlete or not session:
        raise ValueError("Current athlete/session must be set in project_config.yaml")

    run_cfg = dict(cfg)
    # Consistency = compare athlete throws against the athlete's own session template.
    run_cfg["golden_template_athlete"] = athlete
    run_cfg["golden_template_session"] = session
    run_cfg["golden_template_mode"] = "consistency"
    return run_golden_template_viewer(run_cfg)

