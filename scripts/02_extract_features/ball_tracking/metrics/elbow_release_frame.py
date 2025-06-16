import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

elbow_angles = []

def calculate_angle(a, b, c):
    """ Returns the angle at point b (in degrees) between points a and c. """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def find_release_frame(frame, frame_idx, _):
    global elbow_angles

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Use right arm (you can flip this if left-handed)
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)
        elbow_angles.append((frame_idx, angle))

        # Only start looking for release after 5 frames
        if len(elbow_angles) > 5:
            # Find frame with minimum angle so far
            angles = [a for (_, a) in elbow_angles]
            min_idx = np.argmin(angles)
            min_frame = elbow_angles[min_idx][0]

            # Once we're 5+ frames past that, lock it in as the release
            if frame_idx - min_frame > 5:
                return min_frame

    return None
