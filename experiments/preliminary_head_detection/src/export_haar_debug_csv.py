import os
import cv2
import csv


from frame_extractor import extract_frames
from head_detection import detect_head


VIDEO_PATH = "experiments/preliminary_head_detection/raw_videos/IMG_1705 2.MOV"
OUTPUT_CSV = "debug_sequence_frames/debug_haar_sequence/debug_haar_sequence.csv"

def main():
    os.makedirs("debug_sequence_frames", exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "head_y"])

        for frame_index, frame in extract_frames(VIDEO_PATH):
            head_data = detect_head(frame)

            if head_data is None:
                writer.writerow([frame_index, ""])
                continue
    
            x, y, w, h, head_y = head_data
            writer.writerow([frame_index, head_y])
    
    print(f"Saved Haar debug CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()