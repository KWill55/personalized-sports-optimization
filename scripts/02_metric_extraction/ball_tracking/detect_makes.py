#!/usr/bin/env python3
import cv2, math, csv, sys, argparse
import numpy as np
from pathlib import Path
import yaml

# -------------------------------
# Load config (same as GUI)
# -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parents[2]

PROJECT_PATH = BASE_DIR / "project_config.yaml"
SESSION_PATH = BASE_DIR / "session_config.yaml"

with open(PROJECT_PATH, "r") as f:
    CFG1 = yaml.safe_load(f)
with open(SESSION_PATH, "r") as f:
    CFG2 = yaml.safe_load(f)

ATHLETE      = str(CFG1["athlete"])
SESSION      = str(CFG1["session"])
FRAME_WIDTH  = int(CFG1["original_frame_width"])
FRAME_HEIGHT = int(CFG1["original_frame_height"])
FPS_DEFAULT  = float(CFG1.get("player_tracking_fps", 30))

SESSION_INFO = CFG2["athletes"][ATHLETE][SESSION]

UPPER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["upper"]))   # ((x1,y1),(x2,y2))
LOWER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["lower"]))

HSV_LOWER = np.array(SESSION_INFO.get("hsv_ranges", {}).get("lower", [5,100,100]), dtype=np.uint8)
HSV_UPPER = np.array(SESSION_INFO.get("hsv_ranges", {}).get("upper", [30,255,255]), dtype=np.uint8)

AREA_MIN  = float(SESSION_INFO.get("ball_area_px", {}).get("min", 30))
AREA_MAX  = float(SESSION_INFO.get("ball_area_px", {}).get("max", 2000))
CIRC_MIN  = float(SESSION_INFO.get("circularity_min", 0.6))
FILL_MIN  = float(SESSION_INFO.get("fill_ratio_min", 0.6))

# Optional extras (same names the GUI understands)
MIN_MOTION   = int(SESSION_INFO.get("min_motion", SESSION_INFO.get("min_motion_px", 0)))
ERODE_ITERS  = int(SESSION_INFO.get("erode_iters", SESSION_INFO.get("morph", {}).get("erode", 0)))
DILATE_ITERS = int(SESSION_INFO.get("dilate_iters", SESSION_INFO.get("morph", {}).get("dilate", 2)))
BLUR_KSIZE   = int(SESSION_INFO.get("blur_ksize", 1))  # 1 = off

# -------------------------------
# Paths
# -------------------------------
SESSION_DIR  = BASE_DIR / "data" / ATHLETE / SESSION
INPUT_DIR    = SESSION_DIR / "videos" / "ball_tracking" / "raw"
OUTPUT_PATH  = SESSION_DIR / "analysis" / "outcomes.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".m4v"}

# -------------------------------
# Helpers
# -------------------------------
def odd_ksize(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1

def is_inside_rect(pt, tl, br) -> bool:
    if pt is None: return False
    x, y = pt
    x1, y1 = tl; x2, y2 = br
    return (min(x1,x2) <= x <= max(x1,x2)) and (min(y1,y2) <= y <= max(y1,y2))

def is_make(trajectory, upper_box, lower_box) -> bool:
    in_upper = False
    waiting  = False
    for i in range(1, len(trajectory)):
        prev, curr = trajectory[i-1], trajectory[i]
        if curr is None:
            continue
        # Entered upper from above (descending)
        if not in_upper and is_inside_rect(curr, *upper_box):
            if prev and curr[1] > prev[1]:
                in_upper = True
                waiting  = True
            continue
        # After upper, the next visible point must be in lower_box
        if in_upper and waiting:
            if is_inside_rect(curr, *lower_box):
                return True
            elif not is_inside_rect(curr, *upper_box):
                return False
    return False

class BallDetector:
    def __init__(self):
        self.prev_center = None

    def detect(self, frame_bgr):
        # GUI parity: resize, then HSV, then blur/morph, then gates + circ*fill score
        fr = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))

        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        if BLUR_KSIZE > 1:
            hsv = cv2.GaussianBlur(hsv, (odd_ksize(BLUR_KSIZE), odd_ksize(BLUR_KSIZE)), 0)

        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        if ERODE_ITERS > 0:
            mask = cv2.erode(mask, None, iterations=ERODE_ITERS)
        if DILATE_ITERS > 0:
            mask = cv2.dilate(mask, None, iterations=DILATE_ITERS)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score  = -1.0
        best_center = None

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if not (AREA_MIN < area < AREA_MAX):
                continue
            per = cv2.arcLength(cnt, True)
            if per <= 0:
                continue
            circ = 4.0 * math.pi * area / (per * per)

            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r <= 0:
                continue
            fill = float(area) / (math.pi * r * r)

            if circ < CIRC_MIN or fill < FILL_MIN:
                continue

            # (We do not drop low-motion points here; scrubbing semantics don’t apply in batch.)
            score = circ * fill
            if score > best_score:
                best_score  = score
                best_center = (int(x), int(y))

        if best_center is not None:
            self.prev_center = best_center
        return best_center, mask, fr  # return resized frame for optional display

# -------------------------------
# Runner
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", action="store_true", help="Show a quick visual while running")
    args = ap.parse_args()

    vids = [p for p in sorted(INPUT_DIR.iterdir(), key=lambda q: q.name.lower())
            if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if not vids:
        print(f"No videos in {INPUT_DIR}")
        return

    print(f"Processing {len(vids)} clips from {INPUT_DIR}")
    results = {}  # feature_key -> made|miss

    det = BallDetector()

    for i, vp in enumerate(vids, 1):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[{i:02d}] {vp.name}: FAILED TO OPEN, skipping")
            continue

        traj = []
        touched_upper = False
        touched_lower = False

        while True:
            ok, frame = cap.read()
            if not ok: break
            center, mask, fr = det.detect(frame)
            traj.append(center if center is not None else None)

            # track whether we ever touch either region (for warnings)
            if center is not None:
                if is_inside_rect(center, *UPPER_HOOP_REGION): touched_upper = True
                if is_inside_rect(center, *LOWER_HOOP_REGION): touched_lower = True

            if args.display:
                vis = fr.copy()
                if mask is not None:
                    overlay = np.zeros_like(vis); overlay[:] = (0,255,255)
                    vis = np.where(mask[...,None]>0,
                                   cv2.addWeighted(overlay, 0.35, vis, 0.65, 0),
                                   vis)
                cv2.rectangle(vis, UPPER_HOOP_REGION[0], UPPER_HOOP_REGION[1], (255,0,0), 2)
                cv2.rectangle(vis, LOWER_HOOP_REGION[0], LOWER_HOOP_REGION[1], (0,0,255), 2)
                if center is not None:
                    cv2.circle(vis, center, 6, (0,255,0), -1)
                cv2.imshow("detect_makes (preview)", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort early
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Aborted by user.")
                    return

        cap.release()

        verdict = "made" if is_make(traj, UPPER_HOOP_REGION, LOWER_HOOP_REGION) else "miss"
        if not (touched_upper or touched_lower):
            print(f"[{i:02d}] {vp.name}: {verdict.upper()}  (⚠ no contact with hoop regions; check HSV/resize) ")
        else:
            print(f"[{i:02d}] {vp.name}: {verdict.upper()}")

        # Map to your features key (same logic as GUI script you shared)
        base = vp.stem            # e.g., "freethrow001_left"
        core = base.split("_")[0] # e.g., "freethrow001"
        feature_key = f"{core}_angles.csv"
        results[feature_key] = verdict

    # Write outcomes.csv
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing (keep latest overwrite semantics)
    existing = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) >= 2:
                    existing[row[0]] = row[1]

    existing.update(results)

    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "outcome"])
        for k, v in sorted(existing.items()):
            w.writerow([k, v])

    print(f"\nWrote {len(results)} labels → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
