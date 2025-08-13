import cv2
import numpy as np
import csv
from pathlib import Path
import math
import yaml

# =========================
# Config
# =========================

# --- Load YAML files ---
with open(Path("project_config.yaml"), "r") as f:
    config1 = yaml.safe_load(f)

with open(Path("session_config.yaml"), "r") as f:
    config2 = yaml.safe_load(f)

# --- Assign constants from project_config.yaml ---
ATHLETE = config1["athlete"]
SESSION = config1["session"]
FRAME_WIDTH = config1["original_frame_width"]
FRAME_HEIGHT = config1["original_frame_height"]
CROP_SIZE = tuple(config1["crop_size"])
PLAYER_TRACKING_FPS = config1["player_tracking_fps"]
BALL_TRACKING_FPS = config1["ball_tracking_fps"]

# --- Assign constants from session_registry.yaml ---
session_info = config2["athletes"][ATHLETE][SESSION]
UPPER_HOOP_REGION = session_info["hoop_regions"]["upper"]
LOWER_HOOP_REGION = session_info["hoop_regions"]["lower"]
HSV_LOWER = np.array(session_info["hsv_ranges"]["lower"], dtype=np.uint8)
HSV_UPPER = np.array(session_info["hsv_ranges"]["upper"], dtype=np.uint8)
AREA_MIN = session_info["ball_area_px"]["min"]
AREA_MAX = session_info["ball_area_px"]["max"]
CIRC_MIN = session_info["circularity_min"]
FILL_MIN = session_info["fill_ratio_min"]

# Display settings
DISPLAY = False
PRINT_TRAJECTORY = True

# =========================
# Paths 
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]
SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
INPUT_FOLDER = SESSION_DIR / "videos" / "ball_tracking" / "raw"
OUTPUT_PATH = SESSION_DIR / "analysis" / "outcomes.csv"

# =========================
# Global Variables
# =========================

prev_center = [None]

# =========================
# Functions
# =========================

def is_inside_region(pos, region):
    if pos is None:
        return False
    x, y = pos
    (x1, y1), (x2, y2) = region
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

def ball_continues_falling(trajectory, index, frames=3):
    for i in range(index, min(index + frames, len(trajectory) - 1)):
        if trajectory[i+1][1] <= trajectory[i][1]:
            return False
    return True

def detect_ball_center(frame, frame_idx):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (AREA_MIN < area < AREA_MAX):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        fill_ratio = area / (math.pi * radius ** 2)

        if circularity < CIRC_MIN or fill_ratio < FILL_MIN:
            continue

        if prev_center[0] is not None:
            dx = abs(x - prev_center[0][0])
            dy = abs(y - prev_center[0][1])
            if dx < 2 and dy < 2:
                continue

        score = circularity * fill_ratio
        if score > best_score:
            best_score = score
            best_candidate = (int(x), int(y))

    if best_candidate:
        prev_center[0] = best_candidate
    return best_candidate

def is_inside_hoop(pos, hoop_top_left, hoop_bottom_right):
    if pos is None:
        return False
    x, y = pos
    return (
        min(hoop_top_left[0], hoop_bottom_right[0]) <= x <= max(hoop_top_left[0], hoop_bottom_right[0]) and
        min(hoop_top_left[1], hoop_bottom_right[1]) <= y <= max(hoop_top_left[1], hoop_bottom_right[1])
    )

def is_make(trajectory, upper_box, lower_box):
    in_upper = False
    waiting_for_next = False

    for i in range(1, len(trajectory)):
        prev, curr = trajectory[i-1], trajectory[i]

        if curr is None:
            continue

        # Step 1: Entered upper box from above
        if not in_upper and is_inside_hoop(curr, *upper_box):
            if prev and curr[1] > prev[1]:  # descending
                in_upper = True
                waiting_for_next = True
            continue

        # Step 2: After upper entry, wait for next visible point
        if in_upper and waiting_for_next:
            if is_inside_hoop(curr, *lower_box):
                return True  # ✅ Entered lower box → MAKE
            elif not is_inside_hoop(curr, *upper_box):
                return False  # ❌ Reappeared outside boxes → MISS
            # else still in upper box or in-between → keep waiting

    return False

# =========================
# Main Script
# =========================

def main():
    # Collect videos
    video_files = sorted([*INPUT_FOLDER.glob("*.mp4"), *INPUT_FOLDER.glob("*.avi")])
    if not video_files:
        print(f"No .mp4 or .avi videos found in {INPUT_FOLDER}")
        return

    print(f"Loaded {len(video_files)} videos from: {INPUT_FOLDER}")
    for i, vp in enumerate(video_files, 1):
        print(f"  [{i:02d}] {vp.name}")

    results = []
    idx = 0  # current clip index

    while 0 <= idx < len(video_files):
        video_path = video_files[idx]
        print(f"\nOpening clip {idx+1}/{len(video_files)} — {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  ❌ Failed to open; skipping.")
            idx += 1
            continue

        # reset per-clip state
        trajectory = []
        frame_idx = 0
        prev_center[0] = None  # reset tracker memory per clip

        # flags from keypress to jump clips
        go_next = False
        go_prev = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect center
            ball_center = detect_ball_center(frame, frame_idx)
            if ball_center:
                trajectory.append(ball_center)

            # Overlay for display
            if DISPLAY:
                if ball_center:
                    cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)
                for pt in trajectory:
                    cv2.circle(frame, pt, 2, (0, 0, 255), -1)

                # Hoop regions
                cv2.rectangle(frame, UPPER_HOOP_REGION[0], UPPER_HOOP_REGION[1], (255, 0, 0), 2)
                cv2.rectangle(frame, LOWER_HOOP_REGION[0], LOWER_HOOP_REGION[1], (0, 0, 255), 2)

                # Clip header
                header = f"Clip {idx+1}/{len(video_files)} — {video_path.name}"
                cv2.putText(frame, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Ball Tracking", frame)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    # finalize current clip result then quit everything
                    break
                elif key == ord('n'):
                    go_next = True
                    break
                elif key == ord('p'):
                    go_prev = True
                    break

            frame_idx += 1

        cap.release()
        if DISPLAY:
            cv2.destroyAllWindows()

        # Compute result for this clip (even if you jumped away)
        if trajectory:
            verdict = "MAKE" if is_make(trajectory, UPPER_HOOP_REGION, LOWER_HOOP_REGION) else "MISS"
        else:
            verdict = "UNKNOWN"
        results.append((video_path.name, verdict))
        print(f"  Result: {verdict}")

        # Navigation logic
        if go_prev:
            idx = max(0, idx - 1)
        elif go_next:
            idx = min(len(video_files) - 1, idx + 1)
        else:
            # natural advance (end of clip)
            idx += 1

        # Quit if user pressed 'q' (we detect by window already closed + no nav intent)
        # If you want an explicit quit, uncomment:
        # if key == ord('q'): break

    # Save CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "outcome"])  # <-- rename headers to match prep script
        for vid, verdict in results:
            # Construct the feature key to match features.csv exactly.
            # If your features 'file' is literally the angles CSV name:
            #   e.g., freethrow001_angles.csv
            # then derive it here. Adjust this line to YOUR convention.
            base = Path(vid).stem  # e.g., "freethrow001_left"
            # Example mapping: strip camera suffixes and append "_angles.csv"
            # Tweak this to your naming scheme.
            core = base.split("_")[0]  # e.g., "freethrow001"
            feature_key = f"{core}_angles.csv"

            # normalize label text
            if verdict.upper() == "MAKE":
                label = "made"
            elif verdict.upper() == "MISS":
                label = "miss"
            else:
                # Skip UNKNOWN to keep dataset clean
                continue

            writer.writerow([feature_key, label])

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()