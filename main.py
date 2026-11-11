"""Convenience entry point for training, prediction, or exploratory analysis."""

import argparse

from src.exploratory_analysis import run_eda
from src.predict import predict
from src.train import train


def main() -> None:
    """Parse the selected project command and execute it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "eda"])
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "predict":
        predict()
    else:
        print(run_eda())


if __name__ == "__main__":
    main()
