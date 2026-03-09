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
    
    def get_head_and_hip_coordinates(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            head = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_y = (left_hip.y + right_hip.y) / 2

            return head.y, hip_y
        return None





