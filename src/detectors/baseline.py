from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os 
import cv2 

class FallDetectorBaseline:

        def __init__(self, threshold):
            self.threshold = threshold
        # self.duration = duration

        #predict whether a sequence classifies as fall (1) or not (0)
        def predict(self, sequence_path, pose_estimator):

            frame_files = sorted(os.listdir(sequence_path))

            head_y_values = []

            
            for file in frame_files:
                frame_path = os.path.join(sequence_path, file)
                frame = cv2.imread(frame_path) #loads each image into memory

                if frame is None: #if frame is not detected it moves onto the next frame
                    continue

                head = pose_estimator.get_head_coordinates(frame) # gets coordinate of the head

                if head: # if head is detected, then the coordinates are stored in the array
                    _, y = head
                    head_y_values.append(y)

            if not head_y_values:
                return 0

            displacement = max(head_y_values) - min(head_y_values) #measures the distance of the head falling
            print (f"Sequence: {sequence_path}")
            print (f"max head_y_values: ,{max(head_y_values)}  min head_y_values: {min(head_y_values)}")
            print (f"Displacement: ", displacement)

            if displacement > self.threshold:
                return 1 # fall
            else:
                return 0 # ADL

           