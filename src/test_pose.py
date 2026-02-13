import cv2
import os
from pose_estimator import PoseEstimator

SEQUENCE_PATH = "datasets/public/sequences/adl-01"

pose_estimator = PoseEstimator()

frame_files = sorted(os.listdir(SEQUENCE_PATH))

Sequence_length = len(frame_files)

for file in frame_files[:Sequence_length]:
    frame_path = os.path.join(SEQUENCE_PATH, file)
    frame = cv2.imread(frame_path)

    head = pose_estimator.get_head_coordinates(frame)

    if head:
        x, y = head
        print (f"{file} -> Head at (x={x:.3f}, y={y:.3f})")
    else:
        print (f"{file} -> head detection failed")
