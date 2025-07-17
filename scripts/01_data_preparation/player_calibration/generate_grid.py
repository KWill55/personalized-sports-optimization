"""
Title: generate_checkerboard.py

Purpose:
    Generate a printable checkerboard pattern for camera calibration.
    Saves PNG and PDF to the calibration directory for your session.

Features:
    - Customizable inner corners and square size
    - Outputs PDF for exact scaling and PNG for preview
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path
import yaml

# ========================================
# Config
# ========================================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
inner_corners = tuple(cfg["inner_corners"]) # Inner corners (columns, rows)
square_size_cm = cfg["square_size_cm"] # Size of one square in cm
dpi = 300  

# ========================================
# Paths
# ========================================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
calib_dir = session_dir / "calibration"
calib_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Compute checkerboard properties
# ========================================
squares_x = inner_corners[0] + 1
squares_y = inner_corners[1] + 1

cm_to_inch = 1 / 2.54
square_size_inch = square_size_cm * cm_to_inch
width_inch = squares_x * square_size_inch
height_inch = squares_y * square_size_inch

width_px = int(width_inch * dpi)
height_px = int(height_inch * dpi)

print(f"[INFO] Generating checkerboard:")
print(f" - Inner corners: {inner_corners}")
print(f" - Squares: {squares_x} x {squares_y}")
print(f" - Square size: {square_size_cm} cm")
print(f" - Image size: {width_px} x {height_px} px ({width_inch:.2f} in x {height_inch:.2f} in)")

# ========================================
# Generate checkerboard image
# ========================================
checkerboard = np.zeros((height_px, width_px), dtype=np.uint8)
square_size_px = int(square_size_inch * dpi)

for y in range(squares_y):
    for x in range(squares_x):
        if (x + y) % 2 == 0:
            y_start = y * square_size_px
            x_start = x * square_size_px
            checkerboard[y_start:y_start + square_size_px, x_start:x_start + square_size_px] = 255

# ========================================
# Save PNG
# ========================================
png_file = calib_dir / f"checkerboard_{inner_corners[0]}x{inner_corners[1]}_{square_size_cm}cm.png"
cv.imwrite(str(png_file), checkerboard)
print(f"[INFO] Saved PNG: {png_file}")

# ========================================
# Save PDF (accurate size)
# ========================================
pdf_file = calib_dir / f"checkerboard_{inner_corners[0]}x{inner_corners[1]}_{square_size_cm}cm.pdf"
c = canvas.Canvas(str(pdf_file), pagesize=letter)

# Convert to points (1 inch = 72 points)
width_points = width_inch * 72
height_points = height_inch * 72

# Center on page
page_width, page_height = letter
x_offset = (page_width - width_points) / 2
y_offset = (page_height - height_points) / 2

# Place PNG on PDF
c.drawImage(str(png_file), x_offset, y_offset, width_points, height_points)
c.showPage()
c.save()
print(f"[INFO] Saved PDF: {pdf_file}")