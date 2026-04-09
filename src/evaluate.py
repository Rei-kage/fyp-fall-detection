import csv
import os
import sys
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from pose_estimator import PoseEstimator
from detectors.baseline import FallDetectorBaseline
from detectors.temporal_model import FallDetectorTemporal
from detectors.temporal_posture_model import FallDetectorTemporalPosture

CSV_PATH = "datasets/public/metadata.csv"
SEQUENCES_PATH = "datasets/public/sequences"





parser = argparse.ArgumentParser()


parser.add_argument("--model_type", choices=["baseline", "temporal", "temporal_posture"], required=True, help="which fall detection model to use")
parser.add_argument("--split", choices=["dev" , "eval"], required=True, help="dataset split to evaluate")
parser.add_argument("--disp_threshold", type=float, required=True, help="Displacement threshold")
parser.add_argument("--velo_threshold", type=float, default=None, help="Velocity threshold (for temporal model)")
parser.add_argument("--duration_threshold", type=int, default= None, help="Duration threshold (for temporal posture model) ")
parser.add_argument("--posture_threshold", type=float, default=None, help="Posture threshold (for temporal posture model)")
parser.add_argument("--visualise", action="store_true" , help="Show mediapipe window ")



args = parser.parse_args()

model_type = args.model_type
split = args.split
disp_threshold = args.disp_threshold
velo_threshold = args.velo_threshold
duration_threshold = args.duration_threshold
posture_threshold = args.posture_threshold

pose_estimator = PoseEstimator(visualise=args.visualise)


# model = FallDetectorBaseline(threshold=THRESHOLD)







if model_type == "baseline":

    model = FallDetectorBaseline(threshold = disp_threshold)

elif model_type == "temporal":
    if velo_threshold is None:
        raise ValueError(f"Temporal model requires args --velo_threshold")
    
    model = FallDetectorTemporal(disp_threshold = disp_threshold, velo_threshold = velo_threshold)

elif model_type == "temporal_posture":
    if posture_threshold is None:
        raise ValueError(f"Temporal posture model requires args --posture_threshold")
    if velo_threshold is None:
        raise ValueError(f"Temporal posture model requires args --posture_threshold")
    if duration_threshold is None:
        raise ValueError(f"Temporal posture model requires args --duration_threshold")
    
    
    model = FallDetectorTemporalPosture(disp_threshold = disp_threshold, velo_threshold=velo_threshold, duration_threshold = duration_threshold, posture_threshold = posture_threshold)



y_true = []
y_pred = []
per_sequence_rows = [] 

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["split"] == split:

            sequence_name = row["sequence"]
            label = int(row["label"])
            
            sequence_path = os.path.join(SEQUENCES_PATH, sequence_name)
            prediction = model.predict(sequence_path, pose_estimator, sequence_name)

            y_true.append(label)
            y_pred.append(prediction)

            if label == 1 and prediction == 1:
                outcome = "TP"
            elif label == 0 and prediction == 0:
                outcome = "TN"
            elif label == 0 and prediction == 1:
                outcome = "FP"
            else:
                outcome = "FN"
            
            per_sequence_rows.append([
                model_type,
                split,
                disp_threshold,
                velo_threshold if model_type in ["temporal", "temporal_posture"] else "",
                duration_threshold if model_type =="temporal_posture" else "",
                posture_threshold if model_type =="temporal_posture" else "",
                sequence_name,
                label,
                prediction,
                outcome

            ])

    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F1 = 2 * (precision * recall) / (precision + recall)
    


if model_type == "baseline":
    print (f"Evaluation for split: {split} with a diplacement threshold of: {disp_threshold} ")

elif model_type == "temporal":
    print (f"Evaluation for split {split} with a displacement threshold of: {disp_threshold} and a velocity threshold of: {velo_threshold}")



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

fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0


results_path = f"evidence/results/results_{split}.csv"

header = [
                "model",
                "split",
                "disp_threshold",
                "velo_threshold",
                "duration_threshold",
                "posture_threshold",
                "tp",
                "fp",
                "tn", 
                "fn",
                "accuracy", 
                "precision", 
                "recall",
                "F1_score",
                "fnr"
                "fpr"
            ]


file_exists = os.path.isfile(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:

        writer.writerow(header)
             
  

    row = ([
        model_type,
        split,
        disp_threshold,
        velo_threshold if model_type in ["temporal", "temporal_posture"] else "",
        duration_threshold if model_type =="temporal_posture" else "",
        posture_threshold if model_type =="temporal_posture" else "",
        tp,
        fp,
        tn,
        fn,
        accuracy,
        precision,
        recall,
        F1,
        fnr,
        fpr

    ])




    writer.writerow(row)

per_sequence_path = f"evidence/per_sequence/per_sequence_{split}.csv"

per_sequence_header = [
    "model",
    "split",
    "disp_threshold",
    "velo_threshold",
    "duration_threshold",
    "posture_threshold",
    "sequence_name",
    "ground_truth",
    "prediction",
    "outcome"
]

per_sequence_file_exists = os.path.isfile(per_sequence_path)

with open(per_sequence_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not per_sequence_file_exists:
        writer.writerow(per_sequence_header)
    
    writer.writerows(per_sequence_rows)
