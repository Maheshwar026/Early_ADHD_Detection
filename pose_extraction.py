import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_pose_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None

    features = []
    for lm in results.pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features)
