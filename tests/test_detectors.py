import os
import unittest
from unittest.mock import patch
import shutil
import tempfile

from src.detectors.baseline import FallDetectorBaseline
from src.detectors.temporal_model import FallDetectorTemporal
from src.detectors.temporal_posture_model import FallDetectorTemporalPosture

class DummyPoseEstimatorNone:
    def get_head_coordinates(self,frame,sequence):
        return None
    
class DummyPoseEstimatorDisplacement:
    def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.7)]
        self.index = 0

    def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorBelowThreshold:
     def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.3)]
        self.index = 0

     def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
     

class DummyPoseEstimatorThreeFramesOneFail:
     def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.7)]
        self.index = 0

     def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    


class DummyPoseEstimatorTemporalLowVelocity:
    def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.4), (0.5, 0.7)]
        self.index = 0

    def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorTemporalHighVelocity:
    def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.2), (0.5, 0.7)]
        self.index = 0

    def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorTemporalLowDisplacement:
    def __init__(self):
        self.values = [(0.5,0.1), (0.5, 0.2), (0.5, 0.5)]
        self.index = 0

    def get_head_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorSingularFrame:
    def __init__(self):
        self.values = [(0.5,0.1)]
        self.index = 0

    def get_head_coordinates(self, frame, sequence):
        if self.index < len(self.values):   
            value = self.values[self.index]
            self.index += 1
            return value
        return None
    def get_head_and_hip_coordinates(self, frame, sequence):
        if self.index < len(self.values):   
            value = self.values[self.index]
            self.index += 1
            return value
        return None

class DummyPoseEstimatorTemporalPostureFall:
    def __init__(self):
        self.values = [(0.1,0.8),(0.3, 0.8), (0.55, 0.6), (0.7, 0.75)]
        self.index = 0

    def get_head_and_hip_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorTemporalPostureFallFail:
    def __init__(self):
        self.values = [(0.1,0.8),(0.3, 0.8), (0.55, 0.75), (0.7, 0.75)]
        self.index = 0

    def get_head_and_hip_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value

class DummyPoseEstimatorSustainMotionFail:
    def __init__(self):
        self.values = [(0.2,0.8),(0.3, 0.8), (0.55, 0.75), (0.7, 0.75)]
        self.index = 0

    def get_head_and_hip_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
class DummyPoseEstimatorSustainMotionFailOnly:
    def __init__(self):
        self.values = [(0.2,0.8),(0.35, 0.8), (0.6, 0.70), (0.7, 0.75)]
        self.index = 0

    def get_head_and_hip_coordinates(self, frame, sequence):
        value = self.values[self.index]
        self.index += 1
        return value
    
  
class TestDetectors(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        open(os.path.join(self.temp_dir, "frame1.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame2.png"), "a").close()       

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("src.detectors.baseline.cv2.imread", return_value=object())
    def test_baseline_returns_0_when_no_heads_detected(self, mock_imread):
        detector = FallDetectorBaseline(threshold = 0.4)
        pose_estimator = DummyPoseEstimatorNone()

        prediction = detector.predict(self.temp_dir, pose_estimator, "test_sequence")

        self.assertEqual(prediction, 0)


    @patch("src.detectors.baseline.cv2.imread", return_value=object())
    def test_baseline_returns_1_when_displacement_exceeds_threshold(self, mock_imread):
        detector = FallDetectorBaseline(threshold=0.4)
        pose_estimator = DummyPoseEstimatorDisplacement()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 1)

    @patch("src.detectors.baseline.cv2.imread", return_value=object())
    def test_baseline_returns_0_when_displacement_below_threshold(self, mock_imread):
        detector = FallDetectorBaseline(threshold=0.4)
        pose_estimator = DummyPoseEstimatorBelowThreshold()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)

    @patch("src.detectors.baseline.cv2.imread", return_value=object())
    def test_baseline_returns_1_when_none_frames_skipped_and_still_predicts(self, mock_imread):
        mock_imread.side_effect = [object(), None, object()]

        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()

        detector = FallDetectorBaseline(threshold=0.4)
        pose_estimator = DummyPoseEstimatorThreeFramesOneFail()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 1)

    @patch("src.detectors.temporal_model.cv2.imread", return_value=object())
    def test_temporal_model_returns_0_when_displacement_passes_but_velocity_fails(self, mock_imread):
        

        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()

        detector = FallDetectorTemporal(disp_threshold=0.4, velo_threshold= 0.4)
        pose_estimator = DummyPoseEstimatorTemporalLowVelocity()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)


    @patch("src.detectors.temporal_model.cv2.imread", return_value=object())
    def test_temporal_model_returns_1_when_displacement_and_velocity_exceeds_threshold(self, mock_imread):
        

        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()

        detector = FallDetectorTemporal(disp_threshold=0.4, velo_threshold= 0.4)
        pose_estimator = DummyPoseEstimatorTemporalHighVelocity()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 1)

    @patch("src.detectors.temporal_model.cv2.imread", return_value=object())
    def test_temporal_model_returns_0_when_displacement_fails_and_velocity_exceeds_threshold(self, mock_imread):
        

        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()

        detector = FallDetectorTemporal(disp_threshold=0.5, velo_threshold= 0.2)
        pose_estimator = DummyPoseEstimatorTemporalLowDisplacement()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)
    
    @patch("src.detectors.temporal_model.cv2.imread", return_value=object())
    def test_temporal_model_returns_0_when_only_one_head_coordinate_detected(self, mock_imread):
        detector = FallDetectorTemporal(disp_threshold=0.4, velo_threshold= 0.4)
        pose_estimator = DummyPoseEstimatorSingularFrame()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)

     
    @patch("src.detectors.temporal_posture_model.cv2.imread", return_value=object())
    def test_temporal_posture_model_returns_1_when_motion_and_posture_indicate_fall(self, mock_imread):
        detector = FallDetectorTemporalPosture(
            disp_threshold=0.4, 
            velo_threshold= 0.15, 
            duration_threshold=2, 
            posture_threshold=0.1)
        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame4.png"), "a").close()

        pose_estimator = DummyPoseEstimatorTemporalPostureFall()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 1)

    @patch("src.detectors.temporal_posture_model.cv2.imread", return_value=object())
    def test_temporal_posture_model_returns_0_when_motion_passes_but_posture_fails(self, mock_imread):
        detector = FallDetectorTemporalPosture(
            disp_threshold=0.4, 
            velo_threshold= 0.15, 
            duration_threshold=2, 
            posture_threshold=0.1)
        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame4.png"), "a").close()

        pose_estimator = DummyPoseEstimatorTemporalPostureFallFail()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)

    @patch("src.detectors.temporal_posture_model.cv2.imread", return_value=object())
    def test_temporal_posture_model_returns_0_when_Displacement_passes_but_sustained_motion_fails(self, mock_imread):
        detector = FallDetectorTemporalPosture(
            disp_threshold=0.4, 
            velo_threshold= 0.15, 
            duration_threshold=2, 
            posture_threshold=0.1)
        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame4.png"), "a").close()

        pose_estimator = DummyPoseEstimatorSustainMotionFail()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)

    @patch("src.detectors.temporal_posture_model.cv2.imread", return_value=object())
    def test_temporal_posture_model_returns_0_when_only_sustained_motion_fails(self, mock_imread):
        detector = FallDetectorTemporalPosture(
            disp_threshold=0.4, 
            velo_threshold= 0.15, 
            duration_threshold=2, 
            posture_threshold=0.1)
        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame4.png"), "a").close()

        pose_estimator = DummyPoseEstimatorSustainMotionFailOnly()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)

    @patch("src.detectors.temporal_posture_model.cv2.imread", return_value=object())
    def test_temporal_posture_model_returns_0_when_less_than_two_valid_coordinates(self, mock_imread):
        detector = FallDetectorTemporalPosture(
            disp_threshold=0.4, 
            velo_threshold= 0.15, 
            duration_threshold=2, 
            posture_threshold=0.1)
        open(os.path.join(self.temp_dir, "frame3.png"), "a").close()
        open(os.path.join(self.temp_dir, "frame4.png"), "a").close()

        pose_estimator = DummyPoseEstimatorSingularFrame()

        prediction = detector.predict(self.temp_dir,pose_estimator,"test_sequence")

        self.assertEqual(prediction, 0)



if __name__ == "__main__":
    unittest.main()