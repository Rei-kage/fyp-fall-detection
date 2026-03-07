import csv
import os
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from pose_estimator import PoseEstimator
from baseline import FallDetectorBaseline

CSV_PATH = "datasets/public/metadata.csv"
SEQUENCES_PATH = "datasets/public/sequences"

model_thresholds = [0.5]

split = sys.argv[1]

pose_estimator = PoseEstimator()
# model = FallDetectorBaseline(threshold=THRESHOLD)

confusion_matrices = []
accuracy_scores = []
precision_scores = []
recall_scores = []





for threshold in model_thresholds:
    model = FallDetectorBaseline(threshold = threshold)

    y_true = []
    y_pred = []

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == split:

                sequence_name = row["sequence"]
                label = int(row["label"])
                
                sequence_path = os.path.join(SEQUENCES_PATH, sequence_name)
                prediction = model.predict(sequence_path, pose_estimator)

                y_true.append(label)
                y_pred.append(prediction)

    confusion_matrices.append(confusion_matrix(y_true, y_pred))
    accuracy_scores.append(accuracy_score(y_true, y_pred))
    precision_scores.append(precision_score(y_true, y_pred))
    recall_scores.append(recall_score(y_true, y_pred))
    


for results in range(len(model_thresholds)):
    print (f"Evaluation for threshold: {model_thresholds[results]}")

    cm = confusion_matrices[results]

    tn, fp, fn, tp = cm.ravel()

    print (f"Confusion Matrix: ")
    print (cm)
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")
print (f"Accuracy: {accuracy_scores[results]}")
print (f"Recall: {recall_scores[results]}")
print (f"Precision: {precision_scores[results]}")