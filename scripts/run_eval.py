import argparse
import subprocess
import sys

MODEL_COMMANDS = {
    "baseline": [
        "python",
        "src/evaluate.py",
        "--model_type", "baseline",
        "--split", "eval",
        "--disp_threshold", "0.4",
        "--visualise"
    ],

    "temporal": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal",
        "--split", "eval",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--visualise"
    ],

    "temporal_posture": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "eval",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.02",
        "--duration_threshold", "3",
        "--posture_threshold", "0.25",
        "--visualise"
    ],
    
}


def main():
    parser = argparse.ArgumentParser(description = "Run Fall detection models")
    parser.add_argument("model", choices=MODEL_COMMANDS.keys(), help="Model to evaluate")

    args = parser.parse_args()

    command = MODEL_COMMANDS[args.model]

    print(f"Running for model: {args.model}")
    print (" ".join(command))
    print()

    subprocess.run(command, check=True)



if __name__ == "__main__":
    main()