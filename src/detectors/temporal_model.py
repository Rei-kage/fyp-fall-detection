import os 
import cv2 
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

class FallDetectorTemporal:
    
    def __init__(self, disp_threshold, velo_threshold):
        self.disp_threshold = disp_threshold
        self.velo_threshold = velo_threshold

    def predict(self, sequence_path, pose_estimator, sequence_name):

            frame_files = sorted(os.listdir(sequence_path))

            head_y_values = []

            
            for file in frame_files:
                frame_path = os.path.join(sequence_path, file)
                frame = cv2.imread(frame_path) #loads each image into memory

                if frame is None: #if frame is not detected it moves onto the next frame
                    continue

                head = pose_estimator.get_head_coordinates(frame, sequence_name) # gets coordinate of the head

                if head: # if head is detected, then the coordinates are stored in the array
                    _, y = head
                    head_y_values.append(y)

            if not head_y_values:
                return 0
            

            #------------------------------------------------------------------------------------#
            #---------------baseline feature --------------------------------------------------#

            displacement = max(head_y_values) - min(head_y_values) #measures the distance of the head falling
            print (f"Sequence: {sequence_path}")
            print (f"max head_y_values: ,{max(head_y_values)}  min head_y_values: {min(head_y_values)}")
            print (f"Displacement: ", displacement)



            #------------------------------------------------------------------------------------#
            #---------------temporal feature --------------------------------------------------#
            
            velocities = []

            for i in range(len(head_y_values) - 1):
                
                #calculate movement between each frame
                v = head_y_values[i + 1] - head_y_values[i]

                velocities.append(v)
            
            max_velocity = max(velocities) if velocities else 0

            print (f"Sequence: {sequence_path}")
            print(f"Displacement: {displacement}")
            print(f"Max velocity: {max_velocity}")
            print(f"first 10 head y values{head_y_values[:10]}")


                



            if displacement > self.disp_threshold and max_velocity > self.velo_threshold:
                print(f"Sequence: {sequence_path} is a fall")
                return 1 # fall
            else:
                print(f"Sequence: {sequence_path} is an adl")
                return 0 # ADL

