import sys, cv2, math, csv, yaml
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# =========================
# Config loader
# =========================
def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "project_config.yaml").exists():
            return parent
    raise FileNotFoundError("project_config.yaml not found")


def load_configs():
    root = find_repo_root()
    proj_path, sess_path = root / "project_config.yaml", root / "session_config.yaml"
    if not proj_path.exists() or not sess_path.exists():
        raise FileNotFoundError("Missing project_config.yaml or session_config.yaml")

    with open(proj_path) as f:
        project_cfg = yaml.safe_load(f)
    with open(sess_path) as f:
        session_cfg = yaml.safe_load(f)

    ATHLETE = str(project_cfg["athlete"])
    SESSION = str(project_cfg["session"])
    FPS_DEFAULT = float(project_cfg["player_tracking_fps"])
    ball_w, ball_h = tuple(project_cfg.get("uncropped_ball_resolution", [1280, 720]))

    session_info = (
        session_cfg.get("athletes", {})
        .get(ATHLETE, {})
        .get(SESSION)
    )

    if session_info is None:
        print(
            f"[WARNING] Missing session_config entry for athlete='{ATHLETE}', session='{SESSION}'. "
            "Using fallback detection/region defaults."
        )
        upper = ((0, 0), (int(ball_w), int(ball_h * 0.5)))
        lower = ((0, int(ball_h * 0.5)), (int(ball_w), int(ball_h)))
        HSV_LOWER = np.array([0, 0, 0], np.uint8)
        HSV_UPPER = np.array([179, 255, 255], np.uint8)
        AREA_MIN, AREA_MAX = 20, 10000
        CIRC_MIN, FILL_MIN = 0.2, 0.2
    else:
        upper = tuple(map(tuple, session_info["hoop_regions"]["upper"]))
        lower = tuple(map(tuple, session_info["hoop_regions"]["lower"]))
        HSV_LOWER = np.array(session_info["hsv_ranges"]["lower"], np.uint8)
        HSV_UPPER = np.array(session_info["hsv_ranges"]["upper"], np.uint8)
        AREA_MIN, AREA_MAX = session_info["ball_area_px"]["min"], session_info["ball_area_px"]["max"]
        CIRC_MIN, FILL_MIN = session_info["circularity_min"], session_info["fill_ratio_min"]

    base = root
    session_dir = base / "data" / ATHLETE / SESSION
    input_folder = session_dir / "videos" / "ball_tracking" / "raw"
    output_path = session_dir / "analysis" / "outcomes.csv"

    return dict(
        ATHLETE=ATHLETE, SESSION=SESSION, FPS_DEFAULT=FPS_DEFAULT,
        UPPER=upper, LOWER=lower, HSV_LOWER=HSV_LOWER, HSV_UPPER=HSV_UPPER,
        AREA_MIN=AREA_MIN, AREA_MAX=AREA_MAX, CIRC_MIN=CIRC_MIN, FILL_MIN=FILL_MIN,
        INPUT_FOLDER=input_folder, OUTPUT_PATH=output_path
    )

# =========================
# Geometry + Ball Detection
# =========================
def inside(pt, rect):
    if pt is None: return False
    (x,y), ((x1,y1),(x2,y2)) = pt, rect
    return min(x1,x2)<=x<=max(x1,x2) and min(y1,y2)<=y<=max(y1,y2)

def is_make(traj, upper, lower):
    in_upper=False
    for i in range(1,len(traj)):
        prev,curr=traj[i-1],traj[i]
        if curr is None: continue
        if not in_upper and inside(curr,upper) and prev and curr[1]>prev[1]:
            in_upper=True; continue
        if in_upper:
            if inside(curr,lower): return True
            elif not inside(curr,upper): return False
    return False

class BallDetector:
    def __init__(self,cfg):
        self.cfg=cfg; self.prev=None
    def detect(self,frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,self.cfg["HSV_LOWER"],self.cfg["HSV_UPPER"])
        mask=cv2.erode(mask,None,1); mask=cv2.dilate(mask,None,2)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_score=-1
        for c in cnts:
            a=cv2.contourArea(c)
            if not (self.cfg["AREA_MIN"]<a<self.cfg["AREA_MAX"]): continue
            p=cv2.arcLength(c,True)
            if p<=0: continue
            circ=4*math.pi*a/(p*p)
            (x,y),r=cv2.minEnclosingCircle(c)
            if r<=0: continue
            fill=a/(math.pi*r*r)
            if circ<self.cfg["CIRC_MIN"] or fill<self.cfg["FILL_MIN"]: continue
            score=circ*fill
            if score>best_score: best=(int(x),int(y)); best_score=score
        if best: self.prev=best
        return best,mask

# =========================
# GUI
# =========================
class FreeThrowReviewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Free Throw Reviewer (PySide6)")
        self.cfg=load_configs()
        self.detector=BallDetector(self.cfg)
        self.timer=QTimer(); self.timer.timeout.connect(self.next_frame)
        self.playing=False; self.cap=None; self.traj=[]; self.manual=None
        self.labels={}
        self.setFocusPolicy(Qt.StrongFocus)

        # --- Layout ---
        vbox=QVBoxLayout()
        self.clipInfo=QLabel("Clip: - | Label: unknown", alignment=Qt.AlignCenter)
        vbox.addWidget(self.clipInfo)
        self.videoLabel=QLabel(alignment=Qt.AlignCenter)
        vbox.addWidget(self.videoLabel)

        h1=QHBoxLayout()
        for name,fn in [
            ("Load Folder", self.load_folder),
            ("Prev Clip (←)", self.prev_video),
            ("Play/Pause (Space)", self.toggle_play),
            ("Next Clip (→)", self.next_video),
            ("Restart (R)", self.restart)
        ]:
            b=QPushButton(name); b.clicked.connect(fn); h1.addWidget(b)
        vbox.addLayout(h1)

        h2=QHBoxLayout()
        for text,label in [("MAKE (2)","made"),("MISS (1)","miss"),("UNKNOWN","unknown")]:
            b=QPushButton(text); b.clicked.connect(lambda _,l=label:self.set_label(l)); h2.addWidget(b)
        vbox.addLayout(h2)

        self.status=QLabel("Ready."); vbox.addWidget(self.status)
        self.setLayout(vbox)

        # init vars
        self.video_files=[]; self.idx=0
        self.load_folder(self.cfg["INPUT_FOLDER"])

    # ---------- IO ----------
    def load_folder(self, initial=None):
        folder = str(initial)
        dlg = QFileDialog.getExistingDirectory(self, "Select Folder", folder)
        if not dlg: return
        p=Path(dlg)
        vids=sorted([x for x in p.iterdir() if x.suffix.lower() in {".mp4",".avi",".mov"}])
        if not vids:
            QMessageBox.warning(self,"No videos","No supported video files found.")
            return
        self.video_files=vids
        self.idx=0
        self._load_existing_labels()
        self.open_video(self.video_files[0])

    def _clip_key(self, path: Path) -> str:
        return path.stem.split("_")[0]+"_angles.csv"

    def _load_existing_labels(self):
        self.labels={}
        out=self.cfg["OUTPUT_PATH"]
        if not out.exists():
            return
        with open(out,"r",newline="") as f:
            r=csv.reader(f); next(r,None)
            for row in r:
                if len(row)>=2:
                    self.labels[row[0]]=row[1]

    def _current_clip_label(self) -> str:
        if not self.video_files:
            return "unknown"
        key=self._clip_key(self.video_files[self.idx])
        return self.manual if self.manual is not None else self.labels.get(key,"unknown")

    def _refresh_info(self):
        if not self.video_files:
            self.clipInfo.setText("Clip: - | Label: unknown")
            return
        path=self.video_files[self.idx]
        label=self._current_clip_label()
        self.clipInfo.setText(
            f"Clip: {path.name} ({self.idx+1}/{len(self.video_files)}) | Label: {label.upper()}"
        )

    def open_video(self,path:Path):
        if self.cap: self.cap.release()
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            QMessageBox.warning(self,"Error",f"Cannot open {path.name}"); return
        self.traj.clear(); self.detector.prev=None; self.manual=None
        fps=self.cap.get(cv2.CAP_PROP_FPS) or self.cfg["FPS_DEFAULT"]
        self.delay=int(1000/min(fps,60))
        self.status.setText(f"Loaded: {path.name}")
        self._refresh_info()

    # ---------- Controls ----------
    def toggle_play(self):
        if not self.cap: return
        self.playing=not self.playing
        if self.playing: self.timer.start(self.delay)
        else: self.timer.stop()
        self.status.setText("Playing" if self.playing else "Paused")

    def next_video(self):
        if not self.video_files: return
        self.save_result(); self.idx=(self.idx+1)%len(self.video_files)
        self.open_video(self.video_files[self.idx])

    def prev_video(self):
        if not self.video_files: return
        self.save_result(); self.idx=(self.idx-1)%len(self.video_files)
        self.open_video(self.video_files[self.idx])

    def restart(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.traj.clear(); self.manual=None
        self.next_frame()
        self._refresh_info()

    # ---------- Frame processing ----------
    def next_frame(self):
        if not self.cap: return
        ok,frame=self.cap.read()
        if not ok:
            self.timer.stop()
            self.playing=False
            self.status.setText("Clip ended.")
            return
        center,mask=self.detector.detect(frame)
        self.traj.append(center if center is not None else None)
        self.draw_overlay(frame,center)
        self.show_frame(frame)

    def draw_overlay(self,frame,center):
        cv2.rectangle(frame,self.cfg["UPPER"][0],self.cfg["UPPER"][1],(255,0,0),2)
        cv2.rectangle(frame,self.cfg["LOWER"][0],self.cfg["LOWER"][1],(0,0,255),2)
        if center: cv2.circle(frame,center,6,(0,255,0),-1)
        for pt in self.traj:
            if pt: cv2.circle(frame,pt,2,(0,200,255),-1)

    def show_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        img=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(img))

    # ---------- Label + Save ----------
    def set_label(self,label):
        self.manual=label
        self.status.setText(f"Labeled: {label.upper()}")
        self._refresh_info()

    def save_result(self):
        if not self.video_files: return
        path=self.video_files[self.idx]
        label=self.manual or self.labels.get(self._clip_key(path),"unknown")
        out=self.cfg["OUTPUT_PATH"]
        out.parent.mkdir(parents=True,exist_ok=True)
        base=self._clip_key(path)
        existing=dict(self.labels)
        if label!="unknown":
            existing[base]=label
        with open(out,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["file","outcome"])
            for k,v in sorted(existing.items()): w.writerow([k,v])
        self.labels=existing

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self.prev_video()
            return
        if key == Qt.Key_Right:
            self.next_video()
            return
        if key == Qt.Key_1:
            self.set_label("miss")
            return
        if key == Qt.Key_2:
            self.set_label("made")
            return
        if key == Qt.Key_Space:
            self.toggle_play()
            return
        if key == Qt.Key_R:
            self.restart()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.save_result()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

# =========================
# Main
# =========================
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=FreeThrowReviewer()
    w.resize(960,720)
    w.show()
    sys.exit(app.exec())
