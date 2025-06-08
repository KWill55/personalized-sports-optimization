import cv2
import numpy as np
import csv
from pathlib import Path
import math

# =========================
# Configuration Parameters
# =========================

# path parameters
SESSION = "freethrow_tests"
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_FOLDER = BASE_DIR / "data" / SESSION / "angled" / "trimmed"
OUTPUT_PATH = BASE_DIR / "data" / SESSION / "angled" / "freethrow_results.csv"

# hoop regions (created in create_hoop_regions.py)
UPPER_HOOP_REGION = ((1223, 362), (1314, 402))
LOWER_HOOP_REGION = ((1223, 402), (1314, 502))

# HSV color range for detecting the ball (created in tune_hsv.py)
HSV_LOWER = np.array([0, 100, 100])
HSV_UPPER = np.array([5, 255, 255])

# Display settings
DISPLAY = False
PRINT_TRAJECTORY = True

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
        if not (200 < area < 2500):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        fill_ratio = area / (math.pi * radius ** 2)

        if circularity < 0.55 or fill_ratio < 0.65:
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
    results = []

    for video_path in sorted(INPUT_FOLDER.glob("*.mp4")):
        print(f"\nProcessing {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video")
            continue

        trajectory = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ball_center = detect_ball_center(frame, frame_idx)
            if ball_center:
                trajectory.append(ball_center)
                if DISPLAY:
                    cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)

            frame_idx += 1

            if DISPLAY:
                for pt in trajectory:
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)
                cv2.rectangle(frame, UPPER_HOOP_REGION[0], UPPER_HOOP_REGION[1], (255, 0, 0), 2)
                cv2.rectangle(frame, LOWER_HOOP_REGION[0], LOWER_HOOP_REGION[1], (0, 0, 255), 2)
                cv2.imshow("Ball Tracking", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cap.release()
        if DISPLAY:
            cv2.destroyAllWindows()

        result = "MAKE" if is_make(trajectory, UPPER_HOOP_REGION, LOWER_HOOP_REGION) else "MISS"
        results.append((video_path.name, result))
        print(f"  Result: {result}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "result"])
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
