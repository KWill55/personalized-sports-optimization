from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from calibration.calibration_lib import (
    calibrate_extrinsics,
    calibrate_mono_intrinsics,
    collect_mono_detections,
    collect_stereo_detections_combined,
    extrinsics_to_dict,
    intrinsics_to_dict,
    prepare_object_points,
)
from domain.types import Extrinsics, Intrinsics
from utils.io_utils import PROJECT_ROOT, load_config

"""
Responsible for orchestrating calibration subsystem  
"""


class CalibrationNode:
    """
    Runs stereo calibration from captured checkerboard images.

    Outputs under data/<athlete>/<session>/calibration/stereo_calibration:
      - calibration_bundle.json
      - stereo_calib.npz
      - stereo_calib_summary.yaml
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def _paths(self) -> Dict[str, Path]:
        session_dir = PROJECT_ROOT / "data" / self.cfg["athlete"] / self.cfg["session"]
        calibration_dir = session_dir / "calibration"
        calib_images_dir = calibration_dir / "calib_images"

        out_dir = session_dir / "calibration" / "stereo_calibration"
        out_dir.mkdir(parents=True, exist_ok=True)

        return {
            "mono_left": calib_images_dir / "mono_left",
            "mono_right": calib_images_dir / "mono_right",
            "pairs": calib_images_dir / "pairs",
            "out_dir": out_dir,
            "npz": out_dir / "stereo_calib.npz",
            "yaml": out_dir / "stereo_calib_summary.yaml",
            "json": out_dir / "calibration_bundle.json",
        }

    def _save_outputs(
        self,
        intr_left: Intrinsics,
        intr_right: Intrinsics,
        extr: Extrinsics,
        image_size_left: tuple[int, int],
        image_size_right: tuple[int, int],
        image_size_stereo: tuple[int, int],
        *,
        paths: Dict[str, Path],
        checkerboard_size: tuple[int, int],
        square_size_in: float,
        save_bundle: bool,
        save_npz: bool,
        save_summary_yaml: bool,
    ) -> None:
        if save_npz:
            np.savez(
                paths["npz"],
                K1=intr_left.K,
                dist1=intr_left.dist,
                K2=intr_right.K,
                dist2=intr_right.dist,
                R=extr.R,
                T=extr.T,
                E=extr.E,
                F=extr.F,
                P1=extr.P1,
                P2=extr.P2,
                rms_stereo=extr.rms,
                rms_left=intr_left.rms,
                rms_right=intr_right.rms,
            )

        if save_summary_yaml:
            summary = {
                "athlete": self.cfg["athlete"],
                "session": self.cfg["session"],
                "checkerboard": {
                    "inner_corners": [int(checkerboard_size[0]), int(checkerboard_size[1])],
                    "square_size_in": float(square_size_in),
                },
                "intrinsics": {
                    "left": {"rms": float(intr_left.rms)},
                    "right": {"rms": float(intr_right.rms)},
                },
                "extrinsics": {
                    "rms": float(extr.rms),
                    "T_norm": float(np.linalg.norm(extr.T)),
                },
            }
            with paths["yaml"].open("w") as f:
                yaml.safe_dump(summary, f, sort_keys=False)

        if not save_bundle:
            return

        bundle = {
            "schema_version": "1.0",
            "checkerboard": {
                "inner_corners": [int(checkerboard_size[0]), int(checkerboard_size[1])],
                "square_size_in": float(square_size_in),
            },
            "intrinsics": {
                "left": intrinsics_to_dict(intr_left, image_size_left),
                "right": intrinsics_to_dict(intr_right, image_size_right),
            },
            "extrinsics": extrinsics_to_dict(extr, image_size_stereo),
        }
        with paths["json"].open("w") as f:
            json.dump(bundle, f, indent=2)

    def run(
        self,
        *,
        save_bundle: bool = True,
        save_npz: bool = True,
        save_summary_yaml: bool = True,
    ) -> Path:
        checkerboard_size = tuple(self.cfg["inner_corners"])
        square_size_in = float(self.cfg["square_size_in"])
        paths = self._paths()

        obj_template = prepare_object_points(checkerboard_size, square_size_in)

        imgpoints_left_mono, image_size_left = collect_mono_detections(paths["mono_left"], checkerboard_size)
        imgpoints_right_mono, image_size_right = collect_mono_detections(paths["mono_right"], checkerboard_size)

        intr_left, _ = calibrate_mono_intrinsics(obj_template, imgpoints_left_mono, image_size_left)
        intr_right, _ = calibrate_mono_intrinsics(obj_template, imgpoints_right_mono, image_size_right)

        imgpoints_left_stereo, imgpoints_right_stereo, image_size_stereo = collect_stereo_detections_combined(
            paths["pairs"],
            checkerboard_size,
        )
        objpoints_stereo = [obj_template.copy() for _ in range(len(imgpoints_left_stereo))]

        extr = calibrate_extrinsics(
            objpoints_stereo,
            imgpoints_left_stereo,
            imgpoints_right_stereo,
            intr_left,
            intr_right,
            image_size_stereo,
        )

        self._save_outputs(
            intr_left,
            intr_right,
            extr,
            image_size_left,
            image_size_right,
            image_size_stereo,
            paths=paths,
            checkerboard_size=checkerboard_size,
            square_size_in=square_size_in,
            save_bundle=save_bundle,
            save_npz=save_npz,
            save_summary_yaml=save_summary_yaml,
        )

        self.print_report(intr_left, intr_right, extr, paths=paths)
        return paths["out_dir"]

    @staticmethod
    def print_report(
        intr_left: Intrinsics,
        intr_right: Intrinsics,
        extr: Extrinsics,
        *,
        paths: Dict[str, Path],
    ) -> None:
        print("\n================== CALIBRATION REPORT ==================")
        print("INTRINSICS")
        print(f"Left  RMS: {intr_left.rms:.6f}")
        print(f"Right RMS: {intr_right.rms:.6f}")

        print("\nEXTRINSICS")
        print(f"Stereo RMS: {extr.rms:.6f}")
        print(f"Baseline |T|: {float(np.linalg.norm(extr.T)):.6f}")

        print("\nOUTPUTS")
        print(f"NPZ:  {paths['npz']}")
        print(f"YAML: {paths['yaml']}")
        print(f"JSON: {paths['json']}")
        print("========================================================\n")


if __name__ == "__main__":
    CalibrationNode().run()
