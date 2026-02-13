import os
from config import DATASET_PATH, VIDEO_EXTENSIONS

#scans dataset directory and returns list of video filenames
def get_video_files(): 
    video_files = []
    #filters for files ending with allowed formats
    for file in os.listdir(DATASET_PATH):
        if any(file.endswith(ext) for ext in VIDEO_EXTENSIONS):
            video_files.append(file)
    return video_files

if __name__ == "__main__":
    videos = get_video_files()
    print("Videos found:")
    for v in videos:
        print(v)