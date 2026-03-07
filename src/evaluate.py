import csv
import os
import sys
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from pose_estimator import PoseEstimator
from baseline import FallDetectorBaseline

CSV_PATH = "datasets/public/metadata.csv"
SEQUENCES_PATH = "datasets/public/sequences"





parser = argparse.ArgumentParser()

parser.add_argument("--split", choices=["dev" , "eval"], required=True)
parser.add_argument("--threshold", type=float, required=True)

args = parser.parse_args()

split = args.split
threshold = args.threshold



pose_estimator = PoseEstimator()
# model = FallDetectorBaseline(threshold=THRESHOLD)








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

    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    



print (f"Evaluation for split: {split} with a threshold of: {threshold}")



tn, fp, fn, tp = cm.ravel()

print (f"Confusion Matrix: ")
print (cm)
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")
print (f"Accuracy: {accuracy}")
print (f"Recall: {recall}")
print (f"Precision: {precision}")


results_path = f"results/baseline_{split}.csv"

file_exists = os.path.isfile(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "model",
            "split",
            "threshold",
            "tp",
            "fp",
            "tn", 
            "fn",
            "accuracy", 
            "precision", 
            "recall"
        ])

    writer.writerow([
        "baseline",
        split,
        threshold,
        tp,
        fp,
        tn,
        fn,
        accuracy,
        precision,
        recall

    ])