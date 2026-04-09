import pandas as pd
import matplotlib.pyplot as plt

# load csv
df = pd.read_csv("debug_sequence_frames/debug_eval_sequences/debug_fall-20.csv") 

frames = df["frame"]
head = df["head y"]
velocity = df["velocity"]
posture = df["posture"]

plt.figure(figsize=(10,5))

plt.plot(frames, head, label="Head position (y)")
plt.plot(frames, velocity, label = "Velocity")
plt.plot(frames, posture, label = "Posture")

plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Head motion and head to hip range during fall sequence")

plt.legend()
plt.grid()

plt.savefig("sequence_plot_graphs/fall_sequence_fall-20_head_velo_plot.png")
print("Graph saved as fall_sequence_plot.png")