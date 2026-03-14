import os
import cv2
import csv

class FallDetectorTemporalPosture:

    def __init__(self, disp_threshold,velo_threshold, duration_threshold, posture_threshold ):

        self.disp_threshold = disp_threshold
        self.velo_threshold = velo_threshold
        self.duration_threshold = duration_threshold
        self.posture_threshold = posture_threshold
        
 
    def predict(self, sequence_path, pose_estimator, sequence_name):

        frame_files = sorted(os.listdir(sequence_path))

        head_y_values = []
        posture_values = []

        for file in frame_files:

            frame_path = os.path.join(sequence_path, file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            coords = pose_estimator.get_head_and_hip_coordinates(frame, sequence_name)

            if coords:
                
                head_y, hip_y = coords

                head_y_values.append(head_y)

                posture = abs(head_y - hip_y)

                posture_values.append(posture)

        if len(head_y_values) < 2:
            return 0
        
        #------------------------------------------------------------------------------------#
        #---------------baseline threshold --------------------------------------------------#

        displacement = max(head_y_values) - min(head_y_values) #measures the distance of the head falling
        print (f"Sequence: {sequence_path}")
        print (f"max head_y_values: ,{max(head_y_values)}  min head_y_values: {min(head_y_values)}")
        print (f"Displacement: ", displacement)



        #------------------------------------------------------------------------------------#
        #---------------temporal threshold --------------------------------------------------#
            
        velocities = []

        for i in range(len(head_y_values) - 1):
                
            #calculate movement between each frame
            v = max(0, head_y_values[i + 1] - head_y_values[i])

            velocities.append(v)

        consecutive = 0
        sustained_motion = False
        

        for i,v in enumerate(velocities):
            
            if v > self.velo_threshold:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= self.duration_threshold:
                print (f"sustained motion detected")
                sustained_motion = True

                fall_frame_index = i
                break


        #------------------------------------------------------------------------------------#
        #---------------posture threshold --------------------------------------------------#

        debug_path = f"debug_{os.path.basename(sequence_path)}.csv"

       
        if sustained_motion:
            posture_on_fall = posture_values[fall_frame_index]
        else:
            posture_on_fall = None

        

        
        
        if sustained_motion:

            fall_frame_file = frame_files[fall_frame_index]

            fall_frame_path = os.path.join(sequence_path, fall_frame_file)
            print(f"velocity spike index: {fall_frame_index}")
            print(f"fall frame file: {fall_frame_path}")
        

       
      
        
        
            print("Head positions:", head_y_values[fall_frame_index-3:fall_frame_index+3])
            print("Velocities:", velocities[fall_frame_index-3:fall_frame_index+3])
        
        
        if displacement > self.disp_threshold and sustained_motion and  posture_on_fall < self.posture_threshold:
            print(f"Sequence: {sequence_path} is a fall")
            return 1 # fall
        else:
            print(f"Sequence: {sequence_path} is an adl")
            return 0 # ADL