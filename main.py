"""Convenience entry point for training or prediction."""
import argparse
from src.train import train
from src.predict import predict

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train","predict","eda"])
    args=parser.parse_args()
    if args.command=="train": train()
    elif args.command=="predict": predict()
    else:
        from src.exploratory_analysis import run_eda
        print(run_eda())

if __name__=="__main__": main()
