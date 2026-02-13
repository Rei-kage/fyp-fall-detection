import os
import random
import csv

SEQUENCES_PATH = "datasets/public/sequences"
METADATA_PATH = "datasets/public/metadata.csv"

random.seed(42) #reproducible split since its a fixed seed

all_sequences = os.listdir(SEQUENCES_PATH)

#seperate falls and adl

fall = [s for s in all_sequences 
            if s.startswith("fall")
        ]

adl = [s for s in all_sequences 
            if s.startswith("adl")
        ]

random.shuffle(fall)
random.shuffle(adl)

fall_split = int(0.7 * len(fall))
adl_split = int(0.7 * len(adl))

fall_dev = fall[:fall_split]
fall_eval = fall[fall_split:]

adl_dev = adl[:adl_split]
adl_eval = adl[adl_split:]

#write metadata file

with open(METADATA_PATH, "w", newline= "") as f:
    writer = csv.writer(f)
    writer.writerow(["sequence", "label", "split"])

    for seq in fall_dev:
        writer.writerow([seq, 1, "dev"])

    for seq in fall_eval:
        writer.writerow([seq, 1, "eval"])

    for seq in adl_dev:
        writer.writerow([seq, 0, "dev"])

    for seq in adl_eval:
        writer.writerow([seq, 0, "eval"])


print ("Public metadata created successfully")


