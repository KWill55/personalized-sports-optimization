"""
Title: track_ball_arc.py

Purpose:
    - Track the trajectory, arc angles, and possibly power eventually of a basketball during free throws

Prerequisites:
    - video clips of basketball free throws from side view

Output:
    - A plot of the ball's trajectory fitted with a parabola saved to ____

Usage:
    Adjust the folder paths and color thresholds as needed
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

# ========================================
# Parameters
# ========================================

# --- Ball Detection Settings ---
LOWER_ORANGE = np.array([5, 100, 100])    
UPPER_ORANGE = np.array([20, 255, 255])

#TODO put frame sizes up here

# ========================================
# Functions
# ========================================

# --- Parabola Function ---
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

# --- Entry Angle (degrees) ---
def get_entry_angle(a, b, x_at_hoop):
    dy_dx = 2 * a * x_at_hoop + b
    angle_rad = math.atan(dy_dx)
    return math.degrees(angle_rad)


# ========================================
# --- Main Video Processing ---
# ========================================

video_path = "your_video.mp4"
cap = cv2.VideoCapture(video_path)

x_data, y_data = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize if needed
    frame = cv2.resize(frame, (960, 540))

    # Convert to HSV and mask for ball color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume largest contour is ball
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        if radius > 5:  # Filter out noise
            x_data.append(x)
            y_data.append(y)  # y is vertical position in image

            # For visualization
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# --- Fit a parabola ---
x_data = np.array(x_data)
y_data = np.array(y_data)

if len(x_data) >= 3:
    params, _ = curve_fit(parabola, x_data, y_data)
    a, b, c = params

    # Estimate entry angle near hoop (e.g., at highest x-value or where y is lowest)
    x_at_hoop = x_data[np.argmin(y_data)]  # Simplified assumption
    entry_angle = get_entry_angle(a, b, x_at_hoop)
    print(f"Entry angle at hoop: {entry_angle:.2f} degrees")

    # --- Plot ---
    x_range = np.linspace(min(x_data), max(x_data), 200)
    y_fit = parabola(x_range, *params)

    plt.gca().invert_yaxis()
    plt.plot(x_data, y_data, 'ro', label="Ball Positions")
    plt.plot(x_range, y_fit, 'b-', label="Fitted Parabola")
    plt.title("Basketball Trajectory")
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Not enough data to fit parabola.")
