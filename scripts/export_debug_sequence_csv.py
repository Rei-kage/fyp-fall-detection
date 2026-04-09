import os
import csv
import cv2
import argparse
import sys

PROJECT_ROOT = "/home/reihan/FYP"
sys.path.append(PROJECT_ROOT)

from src.pose_estimator import PoseEstimator


SEQUENCE_PATH = "datasets/public/sequences/adl-12"
OUTPUT_CSV = "debug_sequence_frames/debug_eval_sequences/debug_adl-12.csv"

def is_image_file(filename):
    valid_exts = {".png"}
    return os.path.splitext(filename.lower())[1] in valid_exts

def get_sorted_frame_paths(sequence_dir):
    files = [f for f in os.listdir(sequence_dir) if is_image_file(f)]
    files.sort()
    return[os.path.join(sequence_dir, f) for f in files]

def main():
    os.makedirs("debug_sequence_frames", exist_ok = True)

    pose_estimator = PoseEstimator(visualise=False)
    sequence_name = os.path.basename(os.path.normpath(SEQUENCE_PATH))
    frame_paths = get_sorted_frame_paths(SEQUENCE_PATH)

    prev_head_y = None

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "head y", "velocity", "posture"])

        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)

            if frame is None:
                writer.writerow([frame_index, "", "", ""])
                continue


            head = pose_estimator.get_head_coordinates(frame, sequence_name)
            head_and_hip = pose_estimator.get_head_and_hip_coordinates(frame, sequence_name)

            if head is None:
                writer.writerow([frame_index, "", "", ""])
                continue

            head_y = head[1]

            if prev_head_y is None:
                velocity = ""
            else:
                velocity = head_y - prev_head_y
            
            if head_and_hip is None:
                posture = ""
            else:
                head_y_for_posture, hip_y = head_and_hip
                posture = abs(head_y_for_posture - hip_y)

            writer.writerow([frame_index, head_y, velocity, posture])
            prev_head_y = head_y
    
    print(f"Saved MP debug CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()