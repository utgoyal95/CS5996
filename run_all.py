# run_all.py
import subprocess
import sys

def main():
    # 1) Train the DCGAN
    ret = subprocess.call([sys.executable, "train.py"])
    if ret != 0:
        print(f"Training failed with exit code {ret}")
        sys.exit(ret)

    # 2) Evaluate the trained models
    ret = subprocess.call([sys.executable, "eval.py"])
    if ret != 0:
        print(f"Evaluation failed with exit code {ret}")
    sys.exit(ret)

if __name__ == "__main__":
    main()