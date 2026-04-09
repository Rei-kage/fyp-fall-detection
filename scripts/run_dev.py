import argparse
import subprocess
import sys

MODEL_COMMANDS = {
    "baseline1": [
        "python",
        "src/evaluate.py",
        "--model_type", "baseline",
        "--split", "dev",
        "--disp_threshold", "0.2",
        "--visualise"
    ],
    "baseline2": [
        "python",
        "src/evaluate.py",
        "--model_type", "baseline",
        "--split", "dev",
        "--disp_threshold", "0.3",
        "--visualise"
    ],
    "baseline3": [
        "python",
        "src/evaluate.py",
        "--model_type", "baseline",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--visualise"
    ],

    "temporal1": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.015",
        "--visualise"
    ],

    "temporal2": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.02",
        "--visualise"
    ],

    "temporal3": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--visualise"
    ],

    "temporal4": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.03",
        "--visualise"
    ],

    "temporal_posture1": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--duration_threshold", "3",
        "--posture_threshold", "0.15",
        "--visualise"
    ],

    "temporal_posture2": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--duration_threshold", "3",
        "--posture_threshold", "0.2",
        "--visualise"
    ],

    "temporal_posture3": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--duration_threshold", "2",
        "--posture_threshold", "0.25",
        "--visualise"
    ],
    


    "temporal_posture4": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.025",
        "--duration_threshold", "2",
        "--posture_threshold", "0.2",
        "--visualise"
    ],

    "temporal_posture5": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.02",
        "--duration_threshold", "2",
        "--posture_threshold", "0.2",
        "--visualise"
    ],

    "temporal_posture6": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.02",
        "--duration_threshold", "3",
        "--posture_threshold", "0.2",
        "--visualise"
    ],

    "temporal_posture7": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
        "--disp_threshold", "0.4",
        "--velo_threshold", "0.02",
        "--duration_threshold", "2",
        "--posture_threshold", "0.25",
        "--visualise"
    ],

    "temporal_posture7": [
        "python",
        "src/evaluate.py",
        "--model_type", "temporal_posture",
        "--split", "dev",
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