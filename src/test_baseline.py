import os

from pose_estimator import PoseEstimator
from baseline import FallDetectorBaseline

CSV_PATH = "datasets/public/metadata.csv"
SEQUENCE_PATH = "datasets/public/sequences/fall-05"
THRESHOLD = 0.5 

pose_estimator = PoseEstimator()
model = FallDetectorBaseline(threshold= 0.5)

prediction = model.predict(SEQUENCE_PATH, pose_estimator)

if prediction == 1:
    print (f"Prediction: , {prediction}  A fall has been detected")
else:
     print (f"Prediction: , {prediction}  A fall has not been detected")
