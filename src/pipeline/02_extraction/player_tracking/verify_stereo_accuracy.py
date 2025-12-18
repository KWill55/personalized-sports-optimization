import cv2
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# Checkerboard parameters
CHECKER_SIZE = cfg["inner_corners"]   # (cols, rows)
SQUARE_SIZE  = cfg["square_size_in"]  # same units as used in calibration (e.g., mm)

# ==========================================================
# PATHS
# ==========================================================
base_dir    = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
calib_path  = session_dir / "calibration" / "stereo_calibration" / "stereo_calib.npz"
test_dir    = session_dir / "calibration" / "check_test"

stereo_img_path = str(test_dir / "pair_01.png")

# ==========================================================
# HELPER: Split combined stereo image into left/right halves
# ==========================================================
def split_stereo_image(image_path, left_out, right_out, w_half=640):
    """Splits a side-by-side stereo image into left/right images."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"⚠️ Could not load image: {image_path}")

    h, w, _ = img.shape
    if w < 2 * w_half:
        raise ValueError(f"Expected at least {2*w_half}px width, got {w}px")

    left  = img[:, :w_half]
    right = img[:, w_half:2*w_half]

    cv2.imwrite(str(left_out), left)
    cv2.imwrite(str(right_out), right)
    print(f"✅ Saved left/right images: {left_out.name}, {right_out.name}")
    return str(left_out), str(right_out)

# Split stereo image if not already split 
left_img  = test_dir / "left.png"
right_img = test_dir / "right.png"
if not left_img.exists() or not right_img.exists():
    split_stereo_image(stereo_img_path, left_img, right_img)

# ==========================================================
# LOAD CALIBRATION
# ==========================================================
calib = np.load(calib_path)
K1, D1 = calib["K1"], calib["dist1"]
K2, D2 = calib["K2"], calib["dist2"]
R,  T  = calib["R"],  calib["T"].reshape(3, 1)

P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K2 @ np.hstack([R, T])

# ==========================================================
# LOAD IMAGES
# ==========================================================
imgL = cv2.imread(str(left_img), cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(str(right_img), cv2.IMREAD_GRAYSCALE)
if imgL is None or imgR is None:
    raise FileNotFoundError("⚠️ Could not load left/right images in check_test folder")

# ==========================================================
# FIND CHECKERBOARD CORNERS
# ==========================================================
print("[INFO] Detecting checkerboard corners...")
retL, cornersL = cv2.findChessboardCorners(imgL, CHECKER_SIZE)
retR, cornersR = cv2.findChessboardCorners(imgR, CHECKER_SIZE)

if not (retL and retR):
    raise RuntimeError("❌ Checkerboard not detected in one or both images")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria)
cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria)

# ==========================================================
# TRIANGULATE 3D POINTS
# ==========================================================

# Undistort 2D corner points to remove lens distortion (convert to ideal pinhole coordinates)
uL = cv2.undistortPoints(cornersL, K1, D1, P=K1)  # Undistorted left image points
uR = cv2.undistortPoints(cornersR, K2, D2, P=K2)  # Undistorted right image points

# Triangulate corresponding points from left/right views into 3D space (homogeneous coordinates)
Xh = cv2.triangulatePoints(P1, P2, uL, uR)  # homogeneous coordinates
W = Xh[3]

# Convert homogeneous coordinates (projective space) to Cartesian coordinates (real 3D space)
X  = (Xh[:3] / W).T  # cartesian coordinates


# ==========================================================
# PRINT 3D COORDINATES OF OUTER CORNERS
# ==========================================================
cols, rows = CHECKER_SIZE
top_left_idx     = 0
top_right_idx    = cols - 1
bottom_left_idx  = (rows - 1) * cols
bottom_right_idx = rows * cols - 1

corners_3d = {
    "Top-Left":     X[top_left_idx],
    "Top-Right":    X[top_right_idx],
    "Bottom-Left":  X[bottom_left_idx],
    "Bottom-Right": X[bottom_right_idx],
}

print("\n=== 3D Coordinates of Checkerboard Corners (Left Camera Frame of Reference) ===")
# Note: (0,0,0) = optical center of left camera
for name, coord in corners_3d.items():
    print(f"{name:13s}(X,Y,Z):    ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")

# ==========================================================
# PRINT DISTANCES BETWEEN GRID CORNERS
# ==========================================================

def dist(a, b): return np.linalg.norm(a - b) # short cut for euclidean distance 

d_top    = dist(corners_3d["Top-Left"],    corners_3d["Top-Right"])
d_bottom = dist(corners_3d["Bottom-Left"], corners_3d["Bottom-Right"])
d_left   = dist(corners_3d["Top-Left"],    corners_3d["Bottom-Left"])
d_right  = dist(corners_3d["Top-Right"],   corners_3d["Bottom-Right"])
d_diag1  = dist(corners_3d["Top-Left"],    corners_3d["Bottom-Right"])
d_diag2  = dist(corners_3d["Top-Right"],   corners_3d["Bottom-Left"])

# Expected ideal dimensions
expected_w = (cols - 1) * SQUARE_SIZE
expected_h = (rows - 1) * SQUARE_SIZE

print("\n\n=== Distances Between Corners ===")
print(f"Top edge length:       {d_top:.2f} (expected {expected_w:.2f})")
print(f"Bottom edge length:    {d_bottom:.2f} (expected {expected_w:.2f})")
print(f"Left edge length:      {d_left:.2f} (expected {expected_h:.2f})")
print(f"Right edge length:     {d_right:.2f} (expected {expected_h:.2f})")
print("==============================================================")





# TODO: implement below later if desired to visually see the calibration grid 

# ==========================================================
# BUILD GROUND-TRUTH 3D CHECKERBOARD
# ==========================================================
objp = np.zeros((CHECKER_SIZE[0] * CHECKER_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKER_SIZE[0], 0:CHECKER_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


# ==========================================================
# VISUALIZE 3D
# ==========================================================
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(objp[:,0], objp[:,1], objp[:,2], label="Ground Truth", s=20)
ax.scatter(X[:,0], X[:,1], X[:,2], label="Triangulated", s=20)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()
plt.title("Checkerboard 3D Verification")
# plt.show()
