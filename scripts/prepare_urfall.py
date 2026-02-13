import os
import shutil

RAW_PATH = "datasets/public/raw/urfall"
SEQUENCE_PATH = "datasets/public/sequences"

#make sures sequences directory exists
os.makedirs(SEQUENCE_PATH, exist_ok=True)

for folder in os.listdir(RAW_PATH):

    sequence_name = folder.replace("-cam0-rgb", "")

    source_path = os.path.join(RAW_PATH, folder)
    target_path = os.path.join(SEQUENCE_PATH, sequence_name) 

    if os.path.exists(target_path):
        print(f"Skipping {sequence_name}, already exists")
        continue

    shutil.copytree(source_path, target_path)

    print(f"{sequence_name} moved and cleaned successfully")