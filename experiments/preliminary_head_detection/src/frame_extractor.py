import cv2
import os

def extract_frames(video_path):

    
    #create VideoCapture object to open and read video file
    video = cv2.VideoCapture(video_path)

    #validation for video
    if not video.isOpened():
        raise Exception("Video failed to open")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = f"experiments/preliminary_head_detection/fall_dataset/{video_name}"
    os.makedirs(video_folder, exist_ok=True)

    frame_index = 0

    #loops through every frame of a video
    while True:
        flag, frame = video.read()

        if not flag:
            break

        
        cv2.imwrite(f"{video_folder}/frame_{frame_index}.png", frame)
        print("image", frame_index)

        yield frame_index, frame
        frame_index += 1

        

    video.release()