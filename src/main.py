from calibration.calibration_node import CalibrationNode
from calibration.calibration_capture_gui import CalibrationCaptureGui

from recording.recording_node import RecordingNode
from recording.recording_capture_gui import RecordingCaptureGui

from primary_measurements.pose_2d_node import Pose2dNode
from primary_measurements.pose_3d_node import Pose3dNode
from primary_measurements.ball_trajectory_node import BallTrajectoryNode
from primary_measurements.side_pose_node import SidePoseNode
from primary_measurements.verify_detection_coverage_node import VerifyDetectionCoverageNode
from primary_measurements.verify_primary_measurements_node import VerifyPrimaryMeasurementsNode

from preprocessing.alignment_node import AlignmentNode
from preprocessing.crop_freethrows_node import CropFreethrowsNode
from preprocessing.phases_node import PhasesNode
from recording.combine_player_feeds_node import CombinePlayerFeedsNode

from evaluation.golden_template_node import GoldenTemplateNode
from evaluation.consistency_node import ConsistencyNode
from pathlib import Path

import subprocess
import sys

import yaml

from utils.io_utils import load_config
from utils.view_images import collect_images_from_folders, view_images

"""
Purpose: orchestrate entire system

"""

def _find_project_config_path(config_filename: str = "project_config.yaml") -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / config_filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {config_filename}")


def _save_config(cfg: dict, config_path: Path) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _resolve_cfg_path(template_or_path: str, cfg: dict) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _ensure_session_skeleton(cfg: dict) -> None:
    if not cfg.get("athlete") or not cfg.get("session"):
        return
    paths_cfg = cfg.get("paths", {})
    for template in paths_cfg.values():
        path = _resolve_cfg_path(template, cfg)
        # Treat extensions as files; otherwise as directories.
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _list_athletes(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def _list_sessions(data_dir: Path, athlete: str) -> list[str]:
    athlete_dir = data_dir / athlete
    if not athlete_dir.exists():
        return []
    return sorted([p.name for p in athlete_dir.iterdir() if p.is_dir()])


def _pick_from_list(items: list[str], prompt: str) -> str | None:
    if not items:
        return None

    print("")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item}")

    raw = input(f"{prompt} (number or name): ").strip()
    if not raw:
        return None

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(items):
            return items[idx - 1]
        return None

    if raw in items:
        return raw
    return None


def _refresh_node_configs(cfg: dict, nodes: list[object]) -> None:
    for node in nodes:
        try:
            setattr(node, "cfg", cfg)
        except Exception:
            pass


def _run_python_script(script_path: Path, args: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolved = script_path if script_path.is_absolute() else project_root / script_path
    if not resolved.exists():
        print(f"\n[WARNING] Script not found: {resolved}")
        return
    cmd = [sys.executable, str(resolved)]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=False)


def _verify_calibration_summary(cfg: dict) -> None:
    calib_dir = _resolve_cfg_path(cfg["paths"]["stereo_calibration"], cfg)
    npz_path = calib_dir / "stereo_calib.npz"
    yaml_path = calib_dir / "stereo_calib_summary.yaml"
    json_path = calib_dir / "calibration_bundle.json"

    print("\nCalibration Artifacts:")
    print(f"  npz : {npz_path} {'(found)' if npz_path.exists() else '(missing)'}")
    print(f"  yaml: {yaml_path} {'(found)' if yaml_path.exists() else '(missing)'}")
    print(f"  json: {json_path} {'(found)' if json_path.exists() else '(missing)'}")

    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                summary = yaml.safe_load(f) or {}
            intr = summary.get("intrinsics", {})
            extr = summary.get("extrinsics", {})
            print("\nCalibration Summary:")
            if "left" in intr:
                print(f"  left RMS : {intr['left'].get('rms')}")
            if "right" in intr:
                print(f"  right RMS: {intr['right'].get('rms')}")
            print(f"  stereo RMS: {extr.get('rms')}")
            print(f"  baseline |T|: {extr.get('T_norm')}")
        except Exception as exc:
            print(f"\n[WARNING] Could not read summary YAML: {exc}")

    if npz_path.exists():
        try:
            import numpy as np
            calib = np.load(npz_path)
            print("\nNPZ keys:", ", ".join(calib.files))
        except Exception as exc:
            print(f"\n[WARNING] Could not inspect NPZ: {exc}")


def _verify_calibration_images(cfg: dict) -> None:
    left_dir = _resolve_cfg_path(cfg["paths"]["calib_mono_left"], cfg)
    right_dir = _resolve_cfg_path(cfg["paths"]["calib_mono_right"], cfg)
    pair_dir = _resolve_cfg_path(cfg["paths"]["calib_pairs"], cfg)

    img_paths = collect_images_from_folders([left_dir, right_dir, pair_dir], recursive=False)

    if not img_paths:
        print("\n[WARNING] No calibration images found.")
        return

    view_images(img_paths, window_title="Calibration Image Verification", show_parent_in_label=True)


def main():

    print("Welcome to the Freethrow Optimization System!\n")

    config_path = _find_project_config_path()
    cfg = load_config(str(config_path.name))
    data_dir = config_path.parent / "data"

    # =========================================================
    # Create Objects
    # =========================================================
    
    calibration_node = CalibrationNode(cfg=cfg)
    calibration_capture_gui = CalibrationCaptureGui()

    recording_node = RecordingNode(cfg=cfg)
    recording_capture_gui = RecordingCaptureGui()

    pose_2d_node = Pose2dNode(cfg=cfg)
    pose_3d_node = Pose3dNode(cfg=cfg)

    ball_trajectory_node = BallTrajectoryNode(cfg=cfg)
    side_pose_node = SidePoseNode(cfg=cfg)
    verify_detection_coverage_node = VerifyDetectionCoverageNode(cfg=cfg)
    verify_primary_measurements_node = VerifyPrimaryMeasurementsNode(cfg=cfg)

    alignment_node = AlignmentNode(cfg=cfg)
    crop_freethrows_node = CropFreethrowsNode(cfg=cfg)
    phases_node = PhasesNode(cfg=cfg)
    combine_player_feeds_node = CombinePlayerFeedsNode(cfg=cfg)

    grading_node = GoldenTemplateNode(cfg=cfg)
    consistency_node = ConsistencyNode(cfg=cfg)

    config_nodes = [
        calibration_node,
        recording_node,
        pose_2d_node,
        pose_3d_node,
        ball_trajectory_node,
        side_pose_node,
        verify_detection_coverage_node,
        verify_primary_measurements_node,
        alignment_node,
        crop_freethrows_node,
        phases_node,
        combine_player_feeds_node,
        consistency_node,
        grading_node,
    ]


    # =========================================================
    # Repl loop to allow user selection for operation repeatedly
    # =========================================================
    # TODO maybe clear screen after each selection by seeking left the cursor 
    # TODO more options for displaying data too 
    # TODO display what data is available (currently created) for the given session 

    try:
        while True:
            print(f"\n\n\nCurrent session: {cfg.get('session')}")
            print(f"Current Athlete: {cfg.get('athlete')}")

            print("\nFor Possible Selections, type help")
           
            selection = input(" - Enter selection: ")
            command = selection.strip().lower()

            if (command == "record"):
                print("\nRecording new session...")
                _run_python_script(Path("src/recording/recording_node.py"))

            elif (command == "help"):
                print("\nAvailable commands:")

                # TODO option to run entire pipeline assuming data is recorded, other assumption options, etc? 
                
                print("\nPress 'q' to quit") 

                print("\nConfiguration:") 
                print("  - create athlete | change athlete")
                print("  - create session | change session") 

                print("\nShared Utils:") 
                print("  - view cameras")
                print("  - view images")
                print("  - view videos")
                print("  - view project")

                print("\nCalibration:") 
                print("  1. record calibration images | view calibration images") 
                print("  2. calibrate ")
                print("  3. verify calibration")

                print("\nRecording Free Throws:") 
                print("  1. record new") 
                print("  2. combine feeds") 

                print("\nPrimary Measurement Extraction:") 
                print("  - pose 2d")  
                print("  - pose 3d") 
                print("  - ball trajectory")  
                print("  - side pose")
                print("  - label shot outcomes")
                print("  verify primary measurements")

                print("\nPreprocessing:") 
                print("  1. label release | verify release")
                print("  2. align primary measurements | verify alignment")
                print("  3. crop primary measurements | verify cropping")

                print("\nCompute Secondary Measurements:") 
                print("  - compute 3d angles") 
                print("  verify secondary measurements gui")

                print("\nEvaluation:") 
                print("  - summarize data")
                print("  - shot consistency")
                print("  - golden template comparison")
                print("  - real-time feedback gui")

            elif (command == "create athlete"):
                new_athlete = input("Enter new athlete name: ").strip()
                if not new_athlete:
                    print("\n[WARNING] Athlete name cannot be empty.")
                    continue

                athlete_dir = data_dir / new_athlete
                athlete_dir.mkdir(parents=True, exist_ok=True)
                cfg["athlete"] = new_athlete
                cfg["session"] = None
                _ensure_session_skeleton(cfg)
                _save_config(cfg, config_path)
                _refresh_node_configs(cfg, config_nodes)
                print(f"\nSet athlete to '{new_athlete}', cleared session, and updated project config.")

            elif (command == "change athlete"):
                athletes = _list_athletes(data_dir)
                picked = _pick_from_list(athletes, "Choose athlete")
                if not picked:
                    print("\n[WARNING] No valid athlete selected.")
                    continue

                cfg["athlete"] = picked
                cfg["session"] = None
                _ensure_session_skeleton(cfg)
                _save_config(cfg, config_path)
                _refresh_node_configs(cfg, config_nodes)
                print(f"\nSet athlete to '{picked}', cleared session, and updated project config.")

            elif (command == "create session"):
                current_athlete = cfg.get("athlete")
                if not current_athlete:
                    print("\n[WARNING] No current athlete in config.")
                    continue

                new_session = input("Enter new session name: ").strip()
                if not new_session:
                    print("\n[WARNING] Session name cannot be empty.")
                    continue

                session_dir = data_dir / current_athlete / new_session
                session_dir.mkdir(parents=True, exist_ok=True)
                cfg["session"] = new_session
                _ensure_session_skeleton(cfg)
                _save_config(cfg, config_path)
                _refresh_node_configs(cfg, config_nodes)
                print(f"\nSet session to '{new_session}' and updated project config.")

            elif (command == "change session"):
                current_athlete = cfg.get("athlete")
                if not current_athlete:
                    print("\n[WARNING] No current athlete in config.")
                    continue

                sessions = _list_sessions(data_dir, current_athlete)
                picked = _pick_from_list(sessions, f"Choose session for '{current_athlete}'")
                if not picked:
                    print("\n[WARNING] No valid session selected.")
                    continue

                cfg["session"] = picked
                _ensure_session_skeleton(cfg)
                _save_config(cfg, config_path)
                _refresh_node_configs(cfg, config_nodes)
                print(f"\nSet session to '{picked}' and updated project config.")
            
            elif (command == "view cameras"):
                print("\nOpening camera identifier...")
                _run_python_script(Path("src/identify_cameras.py"))

            elif (command == "record calibration images"):
                mode = input("Mode [both/mono/stereo] (default both): ").strip().lower() or "both"
                gui = CalibrationCaptureGui(cfg=cfg)
                if mode in ("both", "all"):
                    gui.run_mono_capture()
                    gui.run_stereo_pair_capture()
                elif mode in ("mono", "left/right"):
                    gui.run_mono_capture()
                elif mode in ("stereo", "pairs", "pair"):
                    gui.run_stereo_pair_capture()
                else:
                    print("\n[WARNING] Unknown mode. Use both/mono/stereo.")

            elif (command in  ("verify calibration images", "view calibration images")):
                _verify_calibration_images(cfg)

            elif (command == "calibrate"):
                print("\nRunning stereo calibration...")
                calibration_node.run()

            elif (command == "verify calibration"):
                _verify_calibration_summary(cfg)
                inspect = input("Open detailed visual inspector? [y/N]: ").strip().lower()
                if inspect in ("y", "yes"):
                    _run_python_script(Path("src/utils/inspect_calibration.py"))

            elif (command == "record new"):
                print("\nLaunching recording GUI...")
                _run_python_script(Path("src/recording/recording_node.py"))

            elif (
                command in (
                    "view data",
                    "view videos",
                    "verify raw",
                    "view raw",
                    "verify/view raw",
                    "view kps 2d",
                    "view keypoints 2d",
                    "view kps_2d",
                    "view trajectories",
                    "view ball trajectories",
                )
            ):
                print("\nOpening video player (select folder in GUI)...")
                _run_python_script(Path("src/utils/video_player.py"))

            elif (command in ("crop freethrows", "crop")):
                print("\nCropping aligned freethrows into fixed release-centered windows (copy only)...")
                crop_freethrows_node.run()

            elif (command in ("verify cropping", "verify crop", "view cropping", "crop viewer")):
                print("\nOpening crop verification viewer...")
                _run_python_script(Path("src/preprocessing/crop_freethrows_node.py"), args=["--view"])

            elif (command in ("label phases", "label phase", "label release")):
                print("\nLaunching phase labeling step...")
                phases_node.run()

            elif (command in ("verify release", "verify phases", "view phases")):
                print("\nOpening phase verification GUI...")
                phases_node.verify()

            elif (command == "align"):
                print("\nRunning release-only alignment...")
                alignment_node.run()

            elif (command in ("verify alignment", "view alignment", "alignment viewer")):
                print("\nOpening alignment verification viewer...")
                _run_python_script(Path("src/preprocessing/alignment_node.py"), args=["--view"])

            elif (command in ("combine feeds", "combine player videos", "combine player feeds")):
                print("\nCombining player feeds...")
                combine_player_feeds_node.run()

            elif (command in ("pose 2d", "extract 2d", "extract player 2d")):
                print("\nRunning 2D pose extraction...")
                pose_2d_node.run()
            
            elif (command in ("pose 3d", "triangulate 3d", "extract player 3d")):
                print("\nRunning 3D triangulation from extracted 2D keypoints...")
                pose_3d_node.triangulate()

            elif (command in ("ball traj", "ball trajectory", "extract ball", "detect ball", "ball detect", "extract trajectories")):
                print("\nRunning ball trajectory extraction pipeline (side camera)...")
                ball_trajectory_node.run()

            elif (command in ("side pose", "extract side pose")):
                print("\nRunning side-camera pose extraction (includes hand keypoints + ROI option)...")
                side_pose_node.run()

            elif (command in ("verify detection coverage", "verify coverage", "detection coverage")):
                print("\nRunning detection coverage verification (2D/3D/ball)...")
                verify_detection_coverage_node.run()

            elif (command in ("label shot outcomes", "label outcomes", "shot outcomes")):
                print("\nOpening shot outcomes labeling GUI...")
                _run_python_script(Path("src/primary_measurements/label_shot_outcomes_gui.py"))

            elif (command in ("verify primary measurements", "verify primary", "primary verify gui")):
                print("\nOpening primary measurements verification GUI...")
                verify_primary_measurements_node.run()

            elif (command in ("compute angles 3d", "compute 3d angles")):
                print("\nComputing 3D-derived kinematics (angles/velocities/accelerations) from existing 3D keypoints...")
                pose_3d_node.compute_secondary()

            elif (command in ("view project", "project gui")):
                print("\nOpening project GUI...")
                _run_python_script(Path("src/utils/project_gui.py"))

            elif (command in ("golden template", "run golden template", "evaluate golden template")):
                compare_athlete = str(cfg.get("athlete", "") or "")
                compare_session = str(cfg.get("session", "") or "")
                print(
                    f"\nOpening golden-template evaluation viewer "
                    f"(compare athlete={compare_athlete}, session={compare_session})..."
                )
                grading_node.run(compare_athlete=compare_athlete, compare_session=compare_session)

            elif (command in ("shot consistency", "consistency", "run consistency")):
                print("\nOpening shot consistency viewer...")
                consistency_node.run()

            elif (command == "q"):
                print("\nClosing program...")
                break

            else:
                print("\n[WARNING] Unknown command. Type help]")

    except KeyboardInterrupt:
        pass
    
    finally:
        try: calibration_node.close()
        except Exception: pass
        try: calibration_capture_gui.close()
        except Exception: pass
        try: recording_node.close()
        except Exception: pass
        try: recording_capture_gui.close()
        except Exception: pass
        try: pose_2d_node.close()
        except Exception: pass
        try: pose_3d_node.close()
        except Exception: pass
        try: ball_trajectory_node.close()
        except Exception: pass
        try: side_pose_node.close()
        except Exception: pass
        try: verify_detection_coverage_node.close()
        except Exception: pass
        try: verify_primary_measurements_node.close()
        except Exception: pass
        try: alignment_node.close()
        except Exception: pass
        try: crop_freethrows_node.close()
        except Exception: pass
        try: phases_node.close()
        except Exception: pass
        try: combine_player_feeds_node.close()
        except Exception: pass
        try: consistency_node.close()
        except Exception: pass
        try: grading_node.close()
        except Exception: pass
    
if __name__ == "__main__":
    main()
