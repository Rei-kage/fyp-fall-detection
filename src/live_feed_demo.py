import cv2
from collections import deque
from pose_estimator import PoseEstimator
import mediapipe as mp

class LiveFeedFallDetector:
# PROOF OF CONCEPT live adapter for the existing temporal-posture model
    def __init__(
            self,
            disp_threshold: float = 0.4,
            velo_threshold: float = 0.02,
            duration_threshold: int = 3,
            posture_threshold: float = 0.25,
            window_size: int = 30,
            cooldown_frames: int = 45,

    ):
        self.disp_threshold = disp_threshold
        self.velo_threshold = velo_threshold
        self.duration_threshold = duration_threshold
        self.posture_threshold = posture_threshold
        self.window_size = window_size
        self.cooldown_frames = cooldown_frames
         
        self.head_y_values = deque(maxlen=window_size)
        self.posture_values = deque(maxlen=window_size)

        self.alert_active = False
        self.alert_cooldown = 0
    
    def update(self, head_y:float, hip_y:float ) -> tuple[bool, dict]:
        posture = abs(head_y - hip_y)

        self.head_y_values.append(head_y)
        self.posture_values.append(posture)

        debug_info = {
            "displacement": 0.0,
            "max_velocity": 0.0,
            "posture_on_fall": None,
            "sustained_motion": False,
        }

        if len(self.head_y_values) < 2:
            return False, debug_info
        
        head_values = list(self.head_y_values)
        posture_values = list(self.posture_values)

        #baseline feature
        displacement = max(head_values) - min(head_values)
        debug_info["displacement"] = displacement

        #Temporal feature
        velocities = []
        for i in range(len(head_values) - 1):
            v = max(0.0, head_values[i + 1] - head_values[i])
            velocities.append(v)

        max_velocity = max(velocities) if velocities else 0.0
        debug_info["max_velocity"] = max_velocity

        consecutive = 0
        sustained_motion = False
        fall_frame_index = None

        for i, v in enumerate(velocities):
            if v > self.velo_threshold:
                consecutive += 1
            else:
                consecutive = 0
            
            if consecutive >= self.duration_threshold:
                sustained_motion = True
                fall_frame_index = i + 1
                break

        debug_info["sustained_motion"] = sustained_motion

        posture_on_fall = None
        if sustained_motion and fall_frame_index is not None and fall_frame_index < len(posture_values):
            posture_on_fall = posture_values[fall_frame_index]

        debug_info["posture_on_fall"] = posture_on_fall


        #cooldown measure
        if self.alert_cooldown > 0:
            self.alert_cooldown -=1
        

        fall_detected = (
            displacement > self.disp_threshold
            and sustained_motion
            and posture_on_fall is not None
            and posture_on_fall < self.posture_threshold
        )


        if fall_detected and self.alert_cooldown == 0:
            self.alert_active = True
            self.alert_cooldown = self.cooldown_frames
        elif self.alert_cooldown == 0:
            self.alert_active = False

        return self.alert_active, debug_info

def draw_overlay(
        frame,
        results,
        mp_drawing,
        mp_pose,
        head_y,
        hip_y,
        status_text,
        debug_info,
    
):

    h, w, _= frame.shape
    if results is not None and results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    
        nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        hx = int (nose_landmark.x * w)
        hy = int(head_y * h)
        hip_pixel = int (hip_y * h)

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
            6,
            (255,0,0),
            -1
        )

    cv2.putText(
        frame,
        "Sequence: Live Feed",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        status_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if "FALL" in status_text else (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Disp: {debug_info['displacement']:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Max Velocity: {debug_info['max_velocity']:.3f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    posture_value = debug_info['posture_on_fall']
    posture_text = "None" if posture_value is None else f"{posture_value:3f}"

    cv2.putText(
        frame,
        f"Posture: {posture_text}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,

    )

    cv2.putText(
        frame,
        f"Sustained: {debug_info['sustained_motion']}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,

    )
    
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not open")
    
    pose_estimator = PoseEstimator(visualise=False)
    detector = LiveFeedFallDetector()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    print("press 'q' to exit live demo")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose_estimator.pose.process(rgb_frame)

        status_text = "NO FALL DETECTED"
        debug_info = {
            "displacement": 0.0,
            "max_velocity": 0.0,
            "posture_on_fall": None,
            "sustained_motion": False,
        }


        head_y = 0.0
        hip_y = 0.0


        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            head = landmarks[mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            head_y = head.y
            hip_y = (left_hip.y + right_hip.y) / 2

            fall_detected, debug_info = detector.update(head_y, hip_y)
            if fall_detected:
                status_text = "FALL DETECTED"

        draw_overlay(
            frame,
            results,
            mp_drawing,
            mp_pose,
            head_y,
            hip_y,
            status_text,
            debug_info

        )
        

        cv2.imshow(f"Pose Tracking", frame)

        if cv2.waitKey(1) & 0xFF ==ord("q"):
         break

if __name__ == "__main__":
    main()


    
    



  