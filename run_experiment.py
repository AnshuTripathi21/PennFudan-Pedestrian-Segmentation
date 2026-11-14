import subprocess
import sys

def run(command):
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

def main():
    python = sys.executable
    common_train = [
        python, "train.py",
        "--epochs", "12",
        "--batch-size", "4",
        "--image-size", "128",
        "--learning-rate", "0.001",
        "--seed", "42",
        "--patience", "4",
        "--workers", "0",
    ]

    for variant in ("baseline", "dilated"):
        run(common_train + ["--variant", variant])
        run([
            python, "evaluate.py",
            "--variant", variant,
            "--image-size", "128",
            "--batch-size", "1",
            "--workers", "0",
            "--warmup", "5",
            "--latency-runs", "20",
            "--save-predictions", "8",
        ])

    print("\nFinished both experiments.")
    print("Check outputs/ for metrics, training logs, checkpoints and prediction images.")

if __name__ == "__main__":
    main()
