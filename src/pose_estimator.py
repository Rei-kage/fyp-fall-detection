import cv2
import mediapipe as mp

#2D pose estimation using MediaPipe 
class PoseEstimator:
    def __init__(self):
        # Load MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def get_head_coordinates(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]

            return nose.x, nose.y
        
        return None


