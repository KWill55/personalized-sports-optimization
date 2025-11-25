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
def load_configs():
    proj_path, sess_path = Path("project_config.yaml"), Path("session_config.yaml")
    if not proj_path.exists() or not sess_path.exists():
        raise FileNotFoundError("Missing project_config.yaml or session_config.yaml")

    with open(proj_path) as f:
        project_cfg = yaml.safe_load(f)
    with open(sess_path) as f:
        session_cfg = yaml.safe_load(f)

    ATHLETE = str(project_cfg["athlete"])
    SESSION = str(project_cfg["session"])
    FPS_DEFAULT = float(project_cfg["player_tracking_fps"])

    SESSION_INFO = session_cfg["athletes"][ATHLETE][SESSION]
    upper = tuple(map(tuple, SESSION_INFO["hoop_regions"]["upper"]))
    lower = tuple(map(tuple, SESSION_INFO["hoop_regions"]["lower"]))
    HSV_LOWER = np.array(SESSION_INFO["hsv_ranges"]["lower"], np.uint8)
    HSV_UPPER = np.array(SESSION_INFO["hsv_ranges"]["upper"], np.uint8)
    AREA_MIN, AREA_MAX = SESSION_INFO["ball_area_px"]["min"], SESSION_INFO["ball_area_px"]["max"]
    CIRC_MIN, FILL_MIN = SESSION_INFO["circularity_min"], SESSION_INFO["fill_ratio_min"]

    base = Path.cwd()
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
        self.playing=False; self.cap=None; self.traj=[]; self.auto=None; self.manual=None

        # --- Layout ---
        vbox=QVBoxLayout()
        self.videoLabel=QLabel(alignment=Qt.AlignCenter)
        vbox.addWidget(self.videoLabel)

        h1=QHBoxLayout()
        for name,fn in [
            ("Load Folder", self.load_folder),
            ("Prev Clip", self.prev_video),
            ("Play/Pause", self.toggle_play),
            ("Next Clip", self.next_video),
            ("Restart", self.restart)
        ]:
            b=QPushButton(name); b.clicked.connect(fn); h1.addWidget(b)
        vbox.addLayout(h1)

        h2=QHBoxLayout()
        for text,label in [("MAKE","made"),("MISS","miss"),("UNKNOWN","unknown")]:
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
        self.video_files=vids; self.idx=0; self.open_video(self.video_files[0])

    def open_video(self,path:Path):
        if self.cap: self.cap.release()
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            QMessageBox.warning(self,"Error",f"Cannot open {path.name}"); return
        self.traj.clear(); self.detector.prev=None; self.auto=self.manual=None
        fps=self.cap.get(cv2.CAP_PROP_FPS) or self.cfg["FPS_DEFAULT"]
        self.delay=int(1000/min(fps,60))
        self.status.setText(f"Loaded: {path.name}")

    # ---------- Controls ----------
    def toggle_play(self):
        if not self.cap: return
        self.playing=not self.playing
        if self.playing: self.timer.start(self.delay)
        else: self.timer.stop()

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
        self.traj.clear(); self.auto=None; self.manual=None
        self.next_frame()

    # ---------- Frame processing ----------
    def next_frame(self):
        if not self.cap: return
        ok,frame=self.cap.read()
        if not ok:
            self.timer.stop(); self.compute_auto(); return
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

    def compute_auto(self):
        if self.auto is None and len(self.traj)>1:
            self.auto="made" if is_make(self.traj,self.cfg["UPPER"],self.cfg["LOWER"]) else "miss"
            self.status.setText(f"Auto: {self.auto}")

    def save_result(self):
        if not self.video_files: return
        path=self.video_files[self.idx]
        label=self.manual or self.auto or "unknown"
        out=self.cfg["OUTPUT_PATH"]
        out.parent.mkdir(parents=True,exist_ok=True)
        base=path.stem.split("_")[0]+"_angles.csv"
        existing={}
        if out.exists():
            with open(out,"r",newline="") as f:
                r=csv.reader(f); next(r,None)
                for row in r:
                    if len(row)>=2: existing[row[0]]=row[1]
        if label!="unknown":
            existing[base]=label
        with open(out,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["file","outcome"])
            for k,v in sorted(existing.items()): w.writerow([k,v])

# =========================
# Main
# =========================
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=FreeThrowReviewer()
    w.resize(960,720)
    w.show()
    sys.exit(app.exec())
