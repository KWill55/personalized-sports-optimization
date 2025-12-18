# annotate_ball_phases_qt.py
"""
Ball Phase Annotator (Video Version, PySide6)
Marks raw_start_frame and raw_ball_through_net for ball-tracking videos.
"""

import sys, csv, yaml, re, cv2
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QStatusBar
)
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
import pandas as pd


def load_configs():
    proj_path = Path("project_config.yaml")
    sess_path = Path("session_config.yaml")
    if not proj_path.exists() or not sess_path.exists():
        raise FileNotFoundError("Missing project_config.yaml or session_config.yaml")

    with open(proj_path, "r") as f:
        project_cfg = yaml.safe_load(f)
    with open(sess_path, "r") as f:
        session_cfg = yaml.safe_load(f)

    ATHLETE = str(project_cfg["athlete"])
    SESSION = str(project_cfg["session"])
    FPS = float(project_cfg["player_tracking_fps"])

    try:
        BASE_DIR = Path(__file__).resolve().parents[2]
    except Exception:
        BASE_DIR = Path.cwd()

    SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION

    app = QApplication.instance() or QApplication([])
    folder = QFileDialog.getExistingDirectory(
        None,
        "Select folder containing ball videos",
        str(SESSION_DIR),
    )
    if not folder:
        raise FileNotFoundError("No folder selected.")
    INPUT_FOLDER = Path(folder)

    OUTPUT_CSV = SESSION_DIR / "metrics" / "ball_phases.csv"
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    return {
        "FPS": FPS,
        "SESSION_DIR": SESSION_DIR,
        "INPUT_FOLDER": INPUT_FOLDER,
        "OUTPUT_CSV": OUTPUT_CSV,
    }


def extract_shot_number(path: Path):
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else float("inf")


class VideoAnnotatorBall(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ball Phase Annotator")
        self.cfg = load_configs()

        self.video_files = sorted(self.cfg["INPUT_FOLDER"].glob("*.avi"), key=extract_shot_number)
        if not self.video_files:
            QMessageBox.critical(self, "No Videos", f"No .avi found in {self.cfg['INPUT_FOLDER']}")
            sys.exit(1)

        self.phase_data = self._load_existing(self.cfg["OUTPUT_CSV"])

        self.idx = 0
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.fps = self.cfg["FPS"]
        self.frame_count = 0
        self.current_frame = 0
        self.current_marks = {
            "raw_start_frame": None,
            "raw_ball_through_net": None,
        }
        self.mark_mode = None

        self._build_ui()
        self._bind_shortcuts()
        self.open_video(self.video_files[self.idx])

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.video_label = QLabel("Video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:black;")
        self.video_label.setMinimumSize(1280, 720)
        layout.addWidget(self.video_label)

        controls = QHBoxLayout()
        self.btn_prev = QPushButton("Previous Clip (left arrow)")
        self.btn_next = QPushButton("Next Clip (right arrow)")
        self.btn_play = QPushButton("Play / Pause (Space)")
        self.btn_restart = QPushButton("Restart")
        self.btn_save = QPushButton("Save (S)")
        self.btn_reload = QPushButton("Reload YAML (L)")
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_next)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_restart)
        controls.addStretch(1)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.btn_reload)
        layout.addLayout(controls)

        frame_controls = QHBoxLayout()
        self.btn_frame_back = QPushButton("Prevous Frame (,)")
        self.btn_frame_fwd = QPushButton("Next Frame (.)")
        frame_controls.addWidget(self.btn_frame_back)
        frame_controls.addWidget(self.btn_frame_fwd)
        layout.addLayout(frame_controls)

        marks = QHBoxLayout()
        self.btn_start = QPushButton("Mark Raw Start (A)")
        self.btn_through = QPushButton("Mark Ball Through Net (N)")
        self.btn_unknown = QPushButton("Mark as Unknown / No Shot (U)")
        self.btn_unknown.clicked.connect(self.mark_unknown)
        layout.addWidget(self.btn_unknown)
        marks.addWidget(self.btn_start)
        marks.addWidget(self.btn_through)
        layout.addLayout(marks)

        self.file_label = QLabel("File: —")
        layout.addWidget(self.file_label)
        self.status = QStatusBar()
        layout.addWidget(self.status)

        self.btn_prev.clicked.connect(self.prev_video)
        self.btn_next.clicked.connect(self.next_video)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_restart.clicked.connect(self.restart_video)
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_reload.clicked.connect(self.reload_yaml)
        self.btn_start.clicked.connect(lambda: self._set_mode("raw_start_frame"))
        self.btn_through.clicked.connect(lambda: self._set_mode("raw_ball_through_net"))
        self.btn_frame_back.clicked.connect(lambda: self.step_frame(-1))
        self.btn_frame_fwd.clicked.connect(lambda: self.step_frame(1))

    def _bind_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_video)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_video)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence("A"), self, activated=lambda: self._set_mode("raw_start_frame"))
        QShortcut(QKeySequence("N"), self, activated=lambda: self._set_mode("raw_ball_through_net"))
        QShortcut(QKeySequence("S"), self, activated=self.save_csv)
        QShortcut(QKeySequence("L"), self, activated=self.reload_yaml)
        QShortcut(QKeySequence(","), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence("."), self, activated=lambda: self.step_frame(1))

    def open_video(self, path: Path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            QMessageBox.warning(self, "OpenCV", f"Could not open {path.name}")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.cfg["FPS"]
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.current_marks = self.phase_data.get(
            path.name,
            {
                "raw_start_frame": None,
                "raw_ball_through_net": None,
            },
        )
        self.file_label.setText(f"File {self.idx+1}/{len(self.video_files)}: {path.name}")
        self.status.showMessage(f"Loaded video ({self.frame_count} frames, {self.fps:.2f} FPS)")
        self.show_frame()

    def show_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
        self.status.showMessage(
            f"Frame {self.current_frame}/{self.frame_count} | Marks: {self.current_marks}"
        )

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        if self.playing:
            self.timer.start(int(1000 / self.fps))
        else:
            self.timer.stop()

    def next_frame(self):
        if not self.cap:
            return
        if self.current_frame >= self.frame_count - 1:
            self.timer.stop()
            self.playing = False
            return
        self.show_frame()

    def step_frame(self, offset: int):
        if not self.cap:
            return
        if self.playing:
            self.playing = False
            self.timer.stop()

        curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if offset < 0:
            new_frame = max(0, curr + offset - 1)
        elif offset > 0:
            new_frame = min(self.frame_count - 1, curr - 1 + offset)
        else:
            new_frame = curr

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.show_frame()
        self.status.showMessage(f"Stepped to frame {new_frame}/{self.frame_count}")

    def restart_video(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.show_frame()

    def next_video(self):
        self.save_csv()
        self.idx = (self.idx + 1) % len(self.video_files)
        self.open_video(self.video_files[self.idx])

    def prev_video(self):
        self.save_csv()
        self.idx = (self.idx - 1) % len(self.video_files)
        self.open_video(self.video_files[self.idx])

    def _set_mode(self, mode):
        self.mark_mode = mode
        self.status.showMessage(f"Mark mode: {mode}. Press ENTER to record current frame.")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.mark_mode and self.cap:
                self.current_marks[self.mark_mode] = self.current_frame
                fname = self.video_files[self.idx].name
                self.phase_data[fname] = self.current_marks.copy()
                self.status.showMessage(f"Set {self.mark_mode} = {self.current_frame}", 3000)
        super().keyPressEvent(event)

    def mark_unknown(self):
        fname = self.video_files[self.idx].name
        if fname in self.phase_data:
            del self.phase_data[fname]
        self.status.showMessage(f"Removed {fname} (marked as UNKNOWN / bad clip)", 3000)

    def save_csv(self):
        rows = []
        for fname, marks in self.phase_data.items():
            if not fname.lower().endswith(".avi"):
                continue
            rows.append([
                fname,
                marks.get("raw_start_frame"),
                marks.get("raw_ball_through_net"),
            ])
        out = self.cfg["OUTPUT_CSV"]
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "raw_start_frame", "raw_ball_through_net"])
            for row in sorted(rows, key=lambda r: r[0]):
                w.writerow(row)
        self.status.showMessage(f"Saved {len(rows)} entries to {out.name}", 3000)

    def reload_yaml(self):
        try:
            new = load_configs()
            self.cfg.update(new)
            self.status.showMessage("Reloaded YAML configs.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "YAML error", str(e))

    def _load_existing(self, path: Path):
        data = {}
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                def safe_int(x):
                    return int(x) if pd.notna(x) else None
                data[row["file"]] = {
                    "raw_start_frame": safe_int(row.get("raw_start_frame")),
                    "raw_ball_through_net": safe_int(row.get("raw_ball_through_net")),
                }
        return data


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = VideoAnnotatorBall()
    w.resize(1440, 900)
    w.show()
    sys.exit(app.exec())
