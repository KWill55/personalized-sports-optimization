import cv2
from pathlib import Path


# =========================
# Configuration Parameters
# =========================

# path paramters
SESSION = "freethrow_tests"
VIDEO_FILE = "freethrow020_trimmed.mp4"
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]
VIDEO_PATH = BASE_DIR / "data" / SESSION / "angled" / "trimmed" / VIDEO_FILE

# hoop region parameters
UPPER_BOX_HEIGHT = 30  
LOWER_BOX_HEIGHT = 80
BOX_PADDING = 1 

# =========================
# Global Variables
# =========================

points = []

# =========================
# Function to Click Rim Points
# =========================

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

# =========================
# Main Function 
# =========================

def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    cv2.namedWindow("Click Rim Left + Right")
    cv2.setMouseCallback("Click Rim Left + Right", click_event)

    while True:
        frame_copy = frame.copy()
        for pt in points:
            cv2.circle(frame_copy, pt, 5, (0, 255, 0), -1)
        cv2.imshow("Click Rim Left + Right", frame_copy)

        if len(points) == 2 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(points) != 2:
        print("⚠️ You must click exactly 2 points: left and right rim")
        return

    # Compute bounding boxes
    rim_left, rim_right = points
    x1, y1 = rim_left
    x2, y2 = rim_right

    top_y = (y1 + y2) // 2

    left = min(x1, x2) - BOX_PADDING
    right = max(x1, x2) + BOX_PADDING
    upper_box = ((left, top_y - UPPER_BOX_HEIGHT), (right, top_y))
    lower_box = ((left, top_y), (right, top_y + LOWER_BOX_HEIGHT))

    print("\nHoop regions for detect_make.py:")
    print(f"UPPER_HOOP_REGION = {upper_box}")
    print(f"LOWER_HOOP_REGION = {lower_box}")

if __name__ == "__main__":
    main()
