#!/usr/bin/env python3
import cv2 as cv, numpy as np, glob, sys
from pathlib import Path
import yaml

# ------------------- config & paths -------------------
cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
project_cfg = yaml.safe_load(open(cfg_path, "r"))

ATHLETE = project_cfg["athlete"]
SESSION = project_cfg["session"]

calib_dir = Path(project_cfg["paths"]["calibration"].format(
    athlete=ATHLETE,
    session=SESSION
))

mono_dir = Path(project_cfg["paths"]["mono_intrinsics"].format(
    athlete=ATHLETE,
    session=SESSION
))

pairs_dir = Path(project_cfg["paths"]["calib_pairs"].format(
    athlete=ATHLETE,
    session=SESSION
))

left_npz  = mono_dir / "intrinsics_left.npz"
right_npz = mono_dir / "intrinsics_right.npz"

# Choose which combined image to use (1280x640). If not given, pick latest.
def pick_pair():
    files = sorted(glob.glob(str(pairs_dir / "pair_*.png")))
    if not files:
        print("[ERROR] No pair_*.png found in", pairs_dir); sys.exit(1)
    return files[-1]

# ------------------- mouse UI -------------------
class ClickUI:
    def __init__(self, imgL, imgR):
        self.L = imgL.copy()
        self.R = imgR.copy()
        self.baseL = imgL.copy()
        self.baseR = imgR.copy()
        self.ptsL, self.ptsR = [], []
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def _draw(self):
        self.L[:] = self.baseL
        self.R[:] = self.baseR
        for i,(x,y) in enumerate(self.ptsL):
            cv.circle(self.L, (int(x),int(y)), 4, (0,255,0), -1)
            cv.putText(self.L, str(i), (int(x)+6,int(y)-6), self.font, 0.5, (0,255,0), 1)
        for i,(x,y) in enumerate(self.ptsR):
            cv.circle(self.R, (int(x),int(y)), 4, (0,255,0), -1)
            cv.putText(self.R, str(i), (int(x)+6,int(y)-6), self.font, 0.5, (0,255,0), 1)
        both = np.hstack([self.L, self.R])
        cv.putText(both, f"Clicks: L={len(self.ptsL)} R={len(self.ptsR)} | "
                         f"Keyboard: [ENTER]=solve  [z]=undo  [c]=clear  [ESC]=quit",
                   (10, 24), self.font, 0.6, (255,255,255), 2)
        cv.imshow("Pick correspondences (left | right)", both)

    def _on_left(self, event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN:
            self.ptsL.append((x,y)); self._draw()
    def _on_right(self, event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN:
            self.ptsR.append((x,y)); self._draw()

    def run(self):
        cv.namedWindow("Pick correspondences (left | right)", cv.WINDOW_NORMAL)
        self._draw()
        # We’ll split the window: left half sends to left handler; right to right handler
        W = self.L.shape[1]
        def on_mouse(event,x,y,flags,param):
            if x < W: self._on_left(event,x,y,flags,param)
            else:     self._on_right(event,x-W,y,flags,param)
        cv.setMouseCallback("Pick correspondences (left | right)", on_mouse)

        while True:
            key = cv.waitKey(20) & 0xFF
            if key == 27:  # ESC
                return None, None
            elif key == ord('z'):
                if self.ptsR: self.ptsR.pop()
                if self.ptsL and (len(self.ptsL) > len(self.ptsR)): self.ptsL.pop()
                self._draw()
            elif key == ord('c'):
                self.ptsL.clear(); self.ptsR.clear(); self._draw()
            elif key == 13 or key == 10:  # ENTER
                n = min(len(self.ptsL), len(self.ptsR))
                if n < 8:
                    print("[WARN] Need at least 8 pairs. Current:", n)
                else:
                    # trim to equal length
                    self.ptsL = self.ptsL[:n]; self.ptsR = self.ptsR[:n]
                    return np.array(self.ptsL, np.float32), np.array(self.ptsR, np.float32)

# ------------------- main solve -------------------
def undistort_norm(xy, K, dist):
    # -> Nx2 normalized points (bearing coords)
    u = cv.undistortPoints(xy.reshape(-1,1,2), K, dist)  # Nx1x2
    return u.reshape(-1,2)

def triangulate_and_reproj_err(P1n, P2n, pts1n, pts2n, K1, dist1, K2, dist2, R, t, pts1_px, pts2_px):
    # triangulate in normalized space, then reprojection error in pixel space
    X4 = cv.triangulatePoints(P1n, P2n, pts1n.T, pts2n.T)  # 4xN
    X = (X4[:3]/X4[3]).T  # Nx3

    # Reproject with distortion to pixel space
    rvec0 = np.zeros(3); tvec0 = np.zeros(3)
    rvec2, _ = cv.Rodrigues(R)
    tvec2 = t.flatten()

    proj1, _ = cv.projectPoints(X, rvec0, tvec0, K1, dist1)  # Nx1x2
    proj2, _ = cv.projectPoints(X, rvec2, tvec2, K2, dist2)
    e1 = np.linalg.norm(proj1.reshape(-1,2) - pts1_px, axis=1)
    e2 = np.linalg.norm(proj2.reshape(-1,2) - pts2_px, axis=1)
    return X, float(np.sqrt(np.mean(e1**2))), float(np.sqrt(np.mean(e2**2)))

def main():
    # --- pick image ---
    combined_path = pick_pair()
    if len(sys.argv) > 1:
        combined_path = sys.argv[1]  # allow CLI override: python ... path/to/pair_XX.png
    baseline_cm = float(sys.argv[2]) if len(sys.argv) > 2 else None

    print("[INFO] Using combined:", combined_path)
    pair = cv.imread(combined_path)
    if pair is None: print("[ERROR] Cannot read", combined_path); sys.exit(1)
    H,W = pair.shape[:2]
    if W < 2: print("[ERROR] Bad image shape."); sys.exit(1)
    mid = W//2
    imgL = pair[:, :mid].copy()
    imgR = pair[:,  mid:].copy()

    # --- load intrinsics ---
    L = np.load(left_npz);  K1, dist1 = L["K"], L["dist"]
    Rz = np.load(right_npz); K2, dist2 = Rz["K"], Rz["dist"]
    print("[INFO] Loaded intrinsics.")

    # --- collect clicks ---
    ui = ClickUI(imgL, imgR)
    ptsL_px, ptsR_px = ui.run()
    cv.destroyAllWindows()
    if ptsL_px is None: 
        print("[INFO] Aborted."); sys.exit(0)

    # --- undistort to normalized coords ---
    ptsL_n = undistort_norm(ptsL_px, K1, dist1)
    ptsR_n = undistort_norm(ptsR_px, K2, dist2)

    # --- essential & pose ---
    # Use normalized points (so focal=1, pp=(0,0))
    E, maskE = cv.findEssentialMat(ptsL_n, ptsR_n, focal=1.0, pp=(0,0), method=cv.RANSAC, prob=0.999, threshold=1e-3)
    if E is None:
        print("[ERROR] findEssentialMat failed."); sys.exit(1)
    inl = int(maskE.sum()) if maskE is not None else len(ptsL_n)
    print(f"[INFO] Inliers after E-RANSAC: {inl}/{len(ptsL_n)}")

    _, R, t, maskPose = cv.recoverPose(E, ptsL_n, ptsR_n, mask=maskE)
    inl_pose = int(maskPose.sum()) if maskPose is not None else inl
    print(f"[INFO] recoverPose inliers: {inl_pose}/{len(ptsL_n)}")
    t = t.reshape(3,1)
    t_norm = float(np.linalg.norm(t))
    print(f"[INFO] Unscaled baseline |t| (arbitrary units): {t_norm:.6f}")

    # Optional: scale translation to requested baseline (cm)
    if baseline_cm is not None and baseline_cm > 0:
        t = t * (baseline_cm / max(t_norm, 1e-9))
        print(f"[INFO] Scaled baseline to {baseline_cm:.2f} cm.")

    # --- build projection matrices (normalized & pixel) ---
    P1n = np.hstack([np.eye(3), np.zeros((3,1))])    # [I|0]
    P2n = np.hstack([R, t])                          # [R|t]

    P1 = K1 @ P1n
    P2 = K2 @ P2n

    # --- F from E ---
    K1_inv = np.linalg.inv(K1); K2_invT = np.linalg.inv(K2).T
    F = K2_invT @ E @ K1_inv

    # --- quick reprojection check on clicked points ---
    X, rmseL, rmseR = triangulate_and_reproj_err(P1n, P2n, ptsL_n, ptsR_n, K1, dist1, K2, dist2, R, t, ptsL_px, ptsR_px)
    print(f"[INFO] Reprojection RMSE (clicked pts): Left={rmseL:.2f}px  Right={rmseR:.2f}px")

    # --- rectification preview ---
    imsize = (imgL.shape[1], imgL.shape[0])  # (W,H)
    R1, R2, P1_rect, P2_rect, Q, _, _ = cv.stereoRectify(K1, dist1, K2, dist2, imsize, R, t)
    map1_L, map2_L = cv.initUndistortRectifyMap(K1, dist1, R1, P1_rect, imsize, cv.CV_16SC2)
    map1_R, map2_R = cv.initUndistortRectifyMap(K2, dist2, R2, P2_rect, imsize, cv.CV_16SC2)
    rectL = cv.remap(imgL, map1_L, map2_L, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, map1_R, map2_R, cv.INTER_LINEAR)
    for y in range(0, rectL.shape[0], 40):
        cv.line(rectL, (0,y), (rectL.shape[1],y), (0,255,0), 1)
        cv.line(rectR, (0,y), (rectR.shape[1],y), (0,255,0), 1)
    cv.imshow("Rectified (L|R)", np.hstack([rectL, rectR]))
    cv.waitKey(0); cv.destroyAllWindows()

    # --- save result ---
    out_dir = calib_dir / "stereo_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stereo_calib_manual.npz"
    np.savez(out_path,
             K1=K1, dist1=dist1, K2=K2, dist2=dist2,
             R=R, T=t, P1=P1, P2=P2, E=E, F=F,
             clicked_L=ptsL_px, clicked_R=ptsR_px,
             used_pair=np.array([combined_path]),
             baseline_cm=np.array([baseline_cm if baseline_cm is not None else -1.0]))
    print("[INFO] Saved:", out_path)
    print("[HINT] If joint angles are your goal, you can leave the baseline unscaled; angles are scale-invariant.")

if __name__ == "__main__":
    main()
