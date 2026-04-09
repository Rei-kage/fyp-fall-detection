import os
import cv2
import argparse
import mediapipe as mp


def is_image_file(filename: str) -> bool:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return os.path.splitext(filename.lower())[1] in valid_exts


def get_sorted_frame_paths(sequence_dir: str):
    files = [f for f in os.listdir(sequence_dir) if is_image_file(f)]
    files.sort()
    return [os.path.join(sequence_dir, f) for f in files]


class SequenceVisualizer:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visualise: bool = True,
    ):
        self.visualise = visualise

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame, sequence_name=""):
        """
        Processes a BGR frame and returns:
        - annotated frame
        - head landmark
        - hip_y midpoint (normalised)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        annotated = frame.copy()
        head = None
        hip_y = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            head = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_y = (left_hip.y + right_hip.y) / 2.0

            if self.visualise:
                h, w, _ = annotated.shape
                hx = int(head.x * w)
                hy = int(head.y * h)
                hip_pixel = int(hip_y * h)

                self.mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                cv2.circle(
                    annotated,
                    (hx, hy),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    annotated,
                    (hx, hip_pixel),
                    6,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated,
                    f"Sequence: {sequence_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

        return annotated, head, hip_y

    def close(self):
        self.pose.close()


def visualise_frame_sequence(sequence_dir, visualizer, output_path=None, fps=10):
    frame_paths = get_sorted_frame_paths(sequence_dir)

    if not frame_paths:
        raise ValueError(f"No image frames found in directory: {sequence_dir}")

    sequence_name = os.path.basename(os.path.normpath(sequence_dir))

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    h, w = first_frame.shape[:2]
    writer = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise ValueError(f"Could not open VideoWriter for: {output_path}")

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARN] Skipping unreadable frame: {frame_path}")
            continue

        annotated, head, hip_y = visualizer.process_frame(
            frame,
            sequence_name=sequence_name
        )

        if visualizer.visualise:
            cv2.imshow("Pose Tracking", annotated)

        if writer is not None:
            annotated = cv2.resize(annotated, (w, h))
            writer.write(annotated)

        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


def visualise_video_file(video_path, visualizer, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 10

    sequence_name = os.path.splitext(os.path.basename(video_path))[0]

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ValueError(f"Could not open VideoWriter for: {output_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, head, hip_y = visualizer.process_frame(
            frame,
            sequence_name=sequence_name
        )

        if visualizer.visualise:
            cv2.imshow("Pose Tracking", annotated)

        if writer is not None:
            annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)

        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise an individual sequence with MediaPipe pose landmarks."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a frame-sequence directory or a video file."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated output video, e.g. outputs/fall20.avi"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Playback FPS for frame-directory sequences."
    )
    parser.add_argument(
        "--model_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model complexity."
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence."
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence."
    )
    parser.add_argument(
        "--no_visualise",
        action="store_true",
        help="Disable live window display."
    )

    args = parser.parse_args()

    visualizer = SequenceVisualizer(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        visualise=not args.no_visualise,
    )

    try:
        if os.path.isdir(args.input):
            visualise_frame_sequence(
                sequence_dir=args.input,
                visualizer=visualizer,
                output_path=args.output,
                fps=args.fps,
            )
        elif os.path.isfile(args.input):
            visualise_video_file(
                video_path=args.input,
                visualizer=visualizer,
                output_path=args.output,
            )
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()