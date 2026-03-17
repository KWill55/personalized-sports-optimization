from calibration.calibration_node import CalibrationNode
from calibration.calibration_capture_gui import CalibrationCaptureGui

from recording.recording_node import RecordingNode
from recording.recording_capture_gui import RecordingCaptureGui

from player_tracking.pose_2d_node import Pose2dNode
from player_tracking.pose_3d_node import Pose3dNode

from ball_tracking.ball_detection_node import BallDetectionNode

from preprocessing.alignment_node import AlignmentNode
from preprocessing.phases_node import PhasesNode
from preprocessing.combine_player_feeds_node import CombinePlayerFeedsNode

from evaluation.grading_node import GradingNode
from pathlib import Path

import subprocess
import sys

import cv2
import yaml

from utils.io_utils import load_config

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


def _run_python_script(script_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolved = script_path if script_path.is_absolute() else project_root / script_path
    if not resolved.exists():
        print(f"\n[WARNING] Script not found: {resolved}")
        return
    subprocess.run([sys.executable, str(resolved)], check=False)


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

    img_paths = (
        sorted(left_dir.glob("*.png"))
        + sorted(right_dir.glob("*.png"))
        + sorted(pair_dir.glob("*.png"))
    )

    if not img_paths:
        print("\n[WARNING] No calibration images found.")
        return

    print(f"\nShowing {len(img_paths)} calibration image(s).")
    print("Controls: any key = next image, ESC = exit")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        preview = img.copy()
        label = f"{img_path.parent.name}/{img_path.name}"
        cv2.putText(preview, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibration Image Verification", preview)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


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

    ball_detection_node = BallDetectionNode(cfg=cfg)

    alignement_node = AlignmentNode(cfg=cfg)
    phases_node = PhasesNode(cfg=cfg)
    combine_player_feeds_node = CombinePlayerFeedsNode(cfg=cfg)

    grading_node = GradingNode(cfg=cfg)

    config_nodes = [
        calibration_node,
        recording_node,
        pose_2d_node,
        pose_3d_node,
        ball_detection_node,
        alignement_node,
        phases_node,
        combine_player_feeds_node,
        grading_node,
    ]


    # =========================================================
    # Repl loop to allow user selection for operation repeatedly
    # =========================================================


    # record calibration images -- gui to record all calibration images 
    # verify calibration images -- gui to view existing 
    # calibrate -- performs intrinsic/extrinsic calibration
    # verify calibration -- shows intrinsic and extrinsic parameters

    # record new 
    # view data

    # combine player videos -- combines player videos in current session if necessary 

    # extract player 2D
    # extract player 3D 
    # extract ball 
    # compute angles 3D

    # draw kps 2D
    # view data
    # draw kps 3D
    # view kps_3D 
    # draw trajectories -- draws ball trajectories
    # view data
    
    # label phases -- TODO later determines windup, release, and followthrough phases for each throw in session
    # verify phases -- TODO later gui to verify phases were aligned properly 
    # trim freethrows -- TODO later eventually have a script to trim freethrows
    # verify trimmed -- TODO later
   
    # align freethrows
    # draw alignment

    # TODO later grade 

    # TODO later summarize data -- gui that summarizes all data for session 
    # TODO later live display -- real time gui (C++) that shows current angles, etc optimal form or not based on data

    # TODO maybe clear screen after each selection by seeking left the cursor 

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
                print("  create athlete") 
                print("  change athlete")
                print("  create session")
                print("  change session")

                print("  view cameras")
                print("  view data")

                print("  record calibration images") # openCV canvas lingers, but ok i guess 
                print("  verify calibration images") # TODO add arrow key functionality and better display words
                print("  calibrate")
                print("  verify calibration")

                print("  record new") # TODO add what free throw it is at top 

                print("  combine feeds") 
                print("  extract 2d")  
                print("  extract 3d") 
                print("  extract trajectories")  
                print("  compute angles 3d") 
                print("  draw kps 2d") 
                print("  draw kps 3d") # does the same as view 3d for now 
                print("  view kps 3d") # does the same as draw 3d for now 
                print("  draw trajectories") # not what i want yet 
                print("  align") # not tested yet 
                print("  draw alignment")
                print("  view project")
                print("  detect ball") # not tested yet 
                print("  q")

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

            elif (command == "verify calibration images"):
                _verify_calibration_images(cfg)

            elif (command == "calibrate"):
                print("\nRunning stereo calibration...")
                calibration_node.run()

            elif (command == "verify calibration"):
                _verify_calibration_summary(cfg)
                inspect = input("Open detailed visual inspector? [y/N]: ").strip().lower()
                if inspect in ("y", "yes"):
                    _run_python_script(Path("src/pipeline/01_collection/player_calibration/inspect_calibration.py"))

            elif (command == "record new"):
                print("\nLaunching recording GUI...")
                _run_python_script(Path("src/recording/recording_node.py"))

            elif (
                command in (
                    "view data",
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

            elif (command == "align"):
                print("\nRunning alignment pipeline...")
                alignement_node.run()

            elif (command in ("combine feeds", "combine player videos", "combine player feeds")):
                print("\nCombining player feeds...")
                combine_player_feeds_node.run()

            elif (command in ("extract 2d", "extract player 2d")):
                print("\nRunning 2D pose extraction...")
                pose_2d_node.run()
            
            elif (command in ("extract 3d", "extract player 3d")):
                print("\nRunning 3D pose reconstruction...")
                pose_3d_node.run()

            elif (command in ("extract ball", "detect ball", "ball detect", "extract trajectories")):
                print("\nRunning ball detection pipeline...")
                ball_detection_node.run()

            elif (command in ("compute angles 3d", "compute 3d angles")):
                print("\nComputing 3D-derived kinematics (angles/velocities/accelerations)...")
                print("Note: this command re-runs the 3D pipeline from existing 2D keypoints.")
                pose_3d_node.run()

            elif (command in ("draw kps 2d", "draw keypoints 2d")):
                _run_python_script(Path("src/pipeline/02_extraction/player_tracking/draw_2d_keypoints.py"))

            elif (command in ("draw kps 3d", "draw keypoints 3d", "view kps 3d", "view keypoints 3d", "view kps_3d")):
                print("\nOpening 3D keypoint viewer...")
                _run_python_script(Path("src/utils/mp33_3d_viewer.py"))

            elif (command in ("draw trajectories", "draw ball trajectories")):
                print("\nOpening trajectory review GUI...")
                _run_python_script(Path("src/pipeline/02_extraction/ball_tracking/detect_makes_gui2.py"))

            elif (command in ("draw alignment", "view alignment")):
                print("\nOpening alignment review GUI...")
                alignement_node.view()

            elif (command in ("view project", "project gui")):
                print("\nOpening project GUI...")
                _run_python_script(Path("src/utils/project_gui.py"))

            elif (command == "q"):
                print("\nClosing program...")
                break

            else:
                print("\n[WARNING] Unknown command. Type help]")
        

            # TODO maybe give options for what the user wants to do or display what data is available
            #   user might only want to open GUIs for instance and they might want to only record more data  
            # switch type statement to see what the user wants to do repeatedly like a repl loop?
            # or boolean values at the top that will say what the user wants to do 
            # probably make a node like UI or UX node for this 

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
        try: ball_detection_node.close()
        except Exception: pass
        try: alignement_node.close()
        except Exception: pass
        try: phases_node.close()
        except Exception: pass
        try: combine_player_feeds_node.close()
        except Exception: pass
        try: grading_node.close()
        except Exception: pass
    
if __name__ == "__main__":
    main()
