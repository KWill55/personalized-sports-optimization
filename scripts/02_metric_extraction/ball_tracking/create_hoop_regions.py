import cv2
import yaml
from pathlib import Path

# =========================
# Configuration Parameters
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]

# --- Load athlete/session from project_config.yaml ---
CONFIG_PATH = BASE_DIR / "project_config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = str(cfg["athlete"])
SESSION = str(cfg["session"])

# Build session_dir AFTER loading athlete/session
SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION

# Where to look for videos
VIDEO_DIR = SESSION_DIR / "videos" / "ball_tracking" / "raw"

# hoop region parameters
UPPER_BOX_HEIGHT = 30
LOWER_BOX_HEIGHT = 80
BOX_PADDING = 1

INSTR = "Click LEFT rim then RIGHT rim | Keys: u=undo, r=reset, Enter=confirm, q=quit"

# =========================
# Find first video file
# =========================
def get_first_video():
    if not VIDEO_DIR.exists():
        return None
    videos = sorted(
        [*VIDEO_DIR.glob("*.mp4"), *VIDEO_DIR.glob("*.avi"),
         *VIDEO_DIR.glob("*.mov"), *VIDEO_DIR.glob("*.mkv")]
    )
    return videos[0] if videos else None

# =========================
# Global Variables
# =========================
points = []  # [(x,y), (x,y)]

# =========================
# Helpers
# =========================
def compute_boxes(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    top_y = (y1 + y2) // 2
    left  = min(x1, x2) - BOX_PADDING
    right = max(x1, x2) + BOX_PADDING
    upper = ((left, top_y - UPPER_BOX_HEIGHT), (right, top_y))
    lower = ((left, top_y), (right, top_y + LOWER_BOX_HEIGHT))
    return upper, lower, top_y

def draw_overlay(img, pts):
    """Draw points, preview boxes (if 2 points), and instructions."""
    vis = img.copy()

    # Instructions banner
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(vis, INSTR, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Draw clicked points
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(vis, f"{i}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # If two points chosen, preview the boxes and the midline
    if len(pts) >= 2:
        upper, lower, top_y = compute_boxes(pts[0], pts[1])
        cv2.line(vis, (0, top_y), (vis.shape[1], top_y), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(vis, upper[0], upper[1], (255, 0, 0), 2)  # upper box (blue-ish)
        cv2.rectangle(vis, lower[0], lower[1], (0, 0, 255), 2)  # lower box (red-ish)
        cv2.putText(vis, "Press Enter to confirm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)

    return vis

# =========================
# Click Event
# =========================
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Allow re-clicking beyond 2: push and keep last two (so you don’t have to reset every time)
        points.append((x, y))
        if len(points) > 2:
            # keep only the last two points
            del points[:-2]
        print(f"Point {len(points)}: ({x}, {y})")

# =========================
# Main
# =========================
def main():
    print(f"athlete={ATHLETE}, session={SESSION}")
    video_path = get_first_video()
    if not video_path:
        print("No video found in session directories.")
        return

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    cv2.namedWindow("Click Rim Left + Right")
    cv2.setMouseCallback("Click Rim Left + Right", click_event)

    confirmed = False
    while True:
        vis = draw_overlay(frame, points)
        cv2.imshow("Click Rim Left + Right", vis)

        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            print("Points reset.")
        elif key == ord('u') and points:
            points.pop()
            print("Undid last point.")
        elif key in (13, 10):  # Enter/Return
            if len(points) == 2:
                confirmed = True
                break

    cv2.destroyAllWindows()
    if not confirmed or len(points) != 2:
        print("Selection not confirmed. Exiting.")
        return

    upper_box, lower_box, _ = compute_boxes(points[0], points[1])

    print("\nHoop regions for detect_make.py:")
    print(f"upper: {list(map(list, upper_box))}")
    print(f"lower: {list(map(list, lower_box))}")


if __name__ == "__main__":
    main()
