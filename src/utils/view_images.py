"""Generic OpenCV image viewer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def close_all_windows() -> None:
    """Best-effort OpenCV window teardown (helps avoid lingering windows on macOS)."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    # Let HighGUI process destroy events.
    for _ in range(3):
        try:
            cv2.waitKey(1)
        except Exception:
            break


def collect_images_from_folders(folders: Iterable[Path], recursive: bool = False) -> list[Path]:
    """Collect image paths from one or more folders."""
    images: list[Path] = []
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            continue
        if recursive:
            candidates = folder.rglob("*")
        else:
            candidates = folder.iterdir()
        for p in candidates:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
    return sorted(images)


def view_images(
    image_paths: list[Path],
    *,
    window_title: str = "Image Viewer",
    show_parent_in_label: bool = True,
) -> int:
    """
    View images sequentially.
    Controls: any key -> next image, ESC -> exit.
    Returns number of displayed images.
    """
    if not image_paths:
        return 0

    shown = 0
    print(f"\nShowing {len(image_paths)} image(s).")
    print("Controls: any key = next image, ESC = exit")

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        preview = img.copy()
        if show_parent_in_label:
            label = f"{image_path.parent.name}/{image_path.name}"
        else:
            label = image_path.name
        cv2.putText(preview, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_title, preview)
        shown += 1
        while True:
            # Break if user closed the window via titlebar.
            visible = cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                close_all_windows()
                return shown
            key = cv2.waitKey(50) & 0xFF
            if key != 255:
                break
        if key == 27:
            break

    close_all_windows()
    return shown
