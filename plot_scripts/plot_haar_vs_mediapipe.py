import pandas as pd
import matplotlib.pyplot as plt

def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return series * 0
    
    return (series - min_val) / (max_val - min_val)


haar_df = pd.read_csv("debug_sequence_frames/debug_haar_sequence/debug_haar_sequence.csv")
mp_df = pd.read_csv("debug_sequence_frames/debug_fall-27.csv")

haar_df["head_y"] = pd.to_numeric(haar_df["head_y"], errors="coerce")
haar_df["detected"] = haar_df["head_y"].notna().astype(int)
mp_df["head_y"] = pd.to_numeric(mp_df["head_y"], errors="coerce")


haar_frames_all = haar_df["frame"]
haar_head_y_all = haar_df["head_y"]

mp_frames_all = mp_df["frame"]
mp_head_y_all = mp_df["head_y"]

haar_valid = haar_df.dropna(subset =["head_y"]).copy()
mp_valid = mp_df.dropna(subset =["head_y"]).copy()

haar_frames = haar_valid["frame"]
haar_head_y = haar_valid["head_y"]

mp_frames = mp_valid["frame"]
mp_head_y = mp_valid["head_y"]

haar_scaled = min_max_scale(haar_head_y)
mp_scaled = min_max_scale(mp_head_y)


fig, axes = plt.subplots(4, 1, figsize=(10,12))

axes[0].scatter(haar_frames, haar_head_y, linewidth=2)
axes[0].set_title("Haar cascade head-position detections (pixel coordinates)")
axes[0].set_xlabel("Frame index")
axes[0].set_ylabel("Head y")
axes[0].grid(True)

axes[1].step(haar_df["frame"], haar_df["detected"], where="mid")
axes[1].set_title("Haar cascade detection availability")
axes[1].set_xlabel("Frame index")
axes[1].set_ylabel("Detection")
axes[1].set_yticks([0,1])
axes[1].grid(True)

axes[2].plot(mp_frames_all, mp_head_y_all, linewidth=2)
axes[2].set_title("MediaPipe head-position estimate (normalised coordinates)")
axes[2].set_xlabel("Frame index")
axes[2].set_ylabel("Head y (normalised)")
axes[2].grid(True)

axes[3].plot(haar_frames, haar_scaled, label ="Haar (rescaled 0-1)", linewidth= 2 )
axes[3].plot(mp_frames, mp_scaled, label = "MediaPipe (rescaled 0-1)", linewidth = 2)
axes[3].set_title("Comparison of head-position traces rescaled")
axes[3].set_xlabel("Frame index")
axes[3].set_ylabel("Relative scale")
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()
plt.savefig("sequence_plot_graphs/haar_vs_mediapipe_comparison.png", dpi=300)
print("saved to sequence_plot_graphs/haar_vs_mediapipe_comparison.png")