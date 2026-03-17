from pathlib import Path
from tkinter import Tk, filedialog
import cv2

"""
Pick a folder and it will tell you the FPS and RES of all videos in that folder 
"""


def pick_folder(initial_dir="."):
    initial_dir = Path(initial_dir).resolve()
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Folder"
    )
    root.destroy()

    if not folder:
        return None

    return Path(folder)


def get_video_recording_info(video_path: Path):
    """
    Returns:
      width, height,
      fps,
      frame_count,
      duration_seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()

    duration = None
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps

    return width, height, fps, frame_count, duration


def main():
    folder = pick_folder(".")
    if folder is None:
        print("Canceled.")
        return

    video_paths = []
    for ext in (".avi", ".mp4"):
        video_paths.extend(folder.rglob(f"*{ext}"))

    if not video_paths:
        print("No videos found.")
        return

    video_paths.sort(key=lambda p: str(p).lower())

    print(f"\nFound {len(video_paths)} video(s) under: {folder}\n")

    header = (
        f"{'Video':60}  "
        f"{'Resolution':12}  "
        f"{'FPS (recorded)':14}  "
        f"{'Frames':8}  "
        f"{'Duration (s)':12}"
    )
    print(header)
    print("-" * len(header))

    for vp in video_paths:
        rel = vp.relative_to(folder)
        info = get_video_recording_info(vp)

        if info is None:
            print(f"{str(rel)[:60]:60}  {'(failed)':12}")
            continue

        w, h, fps, n, dur = info

        res = f"{w}x{h}" if w and h else "unknown"
        fps_str = f"{fps:.3f}" if fps else "unknown"
        frames_str = str(n) if n else "unknown"
        dur_str = f"{dur:.3f}" if dur else "unknown"

        print(
            f"{str(rel)[:60]:60}  "
            f"{res:12}  "
            f"{fps_str:14}  "
            f"{frames_str:8}  "
            f"{dur_str:12}"
        )


if __name__ == "__main__":
    main()
