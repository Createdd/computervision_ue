import argparse
import os
from utils import create_detection_dict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


parser = argparse.ArgumentParser(
    description="Get detections for the specified data set.")
parser.add_argument('--set', type=str, default="validation",
                    help="Defines the sample set and must be " +
                         "in {train, validation, test}")
parser.add_argument('--dataset_root', type=str, default=os.path.join('..','data'),
                    help="Location of the dataset.")


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.set in ["train", "validation", "test"]
    assert args.set in os.listdir(args.dataset_root)

    create_detection_dict(args.set, args.dataset_root)
        