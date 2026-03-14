import cv2
import mediapipe as mp

#2D pose estimation using MediaPipe 
class PoseEstimator:
    def __init__(self, visualise = False):
        # Load MediaPipe Pose model

        self.visualise = visualise
        self.mp_pose = mp.solutions.pose

        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_head_coordinates(self, frame, sequence_name=None):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]

            h,w,_ = frame.shape
            x = int(nose.x * w)
            y = int(nose.y * h)

            if self.visualise:

                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            
                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0,0,255),
                    -1
                )

                cv2.imshow(f"Pose Tracking {sequence_name}", frame)

                if cv2.waitKey(1) & 0xFF ==ord("q"):
                    return None
        
            return nose.x, nose.y
    
        return None
    
    def get_head_and_hip_coordinates(self, frame, sequence_name=None):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            head = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_y = (left_hip.y + right_hip.y) / 2

            if self.visualise:
                h, w, _ = frame.shape
                hx = int(head.x * w)
                hy = int(head.y * h)
                hip_pixel = int(hip_y * h)

                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                cv2.circle(
                    frame,
                    (hx, hy),
                    6,
                    (0,0,255),
                    -1
                )

                cv2.circle(
                    frame,
                    (hx, hip_pixel),
                    6,(255,0,0),
                    -1
                )
                cv2.putText(
                    frame,
                    f"Sequence: {sequence_name}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )
                cv2.imshow(f"Pose Tracking", frame)

                if cv2.waitKey(1) & 0xFF ==ord("q"):
                    return None
                        

            return head.y, hip_y
        
        return None





