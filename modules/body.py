"""
Fetching and processing of points on the user's body in 3D space
"""

import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_body_points(frame):
    """Gets 3D points from the camera feed"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    return results.pose_landmarks


def render_body(frame, results):
    """Shows points on the camera feed and returns the new frame"""
    for landmark_id, lm in enumerate(results.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y*h)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
    return frame
