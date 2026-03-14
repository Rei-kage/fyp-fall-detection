from frame_extractor import extract_frames
from head_detection import detect_head
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


VIDEO_PATH = "experiments/preliminary_head_detection/raw_videos/IMG_1704.MOV"

def main():
    frame_numbers = []
    head_positions = []
    Video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    print(f"Expected frames: {int(cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FRAME_COUNT))} ")

    for frame_index, frame in extract_frames(VIDEO_PATH):
        

        result = detect_head(frame)

        head_y_value = np.nan

        if result is not None:

            x, y, w, h, head_y = result
            head_y_value = head_y
            print("Frame:", frame_index, "Head Y:", head_y)
            
        
            cv2.rectangle(
                frame,
                (x,y),
                (x + w, y + h),
                (0,255,0),
                2
            )

            cv2.circle(
                frame,
                (x + w//2, head_y),
                5,
                (255,0,0),
                -1
            )

        head_positions.append(head_y_value)
        frame_numbers.append(frame_index)
       

        cv2.imshow("Head Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    plt.grid(True)
    plt.plot(frame_numbers, head_positions)
    plt.xlabel("Frame")
    plt.ylabel("Head Y Position")
    plt.title(f"Head Position for sequence {Video_name}")
    plt.savefig(f"experiments/head_positions/{Video_name}_trajectory.png")

      


if __name__ == "__main__":
    main()