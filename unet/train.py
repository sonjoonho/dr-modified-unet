import argparse
from pathlib import Path

from unet.data import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="../data/raw/IDRiD", help="Directory with the IDRiD dataset"
)
parser.add_argument("--batch_size", default=16, help="Batch size")


def main():
    args = parser.parse_args()

    batch_size: int = args.batch_size
    data_dir: Path = Path(args.data_dir).resolve()

    train_dataset, test_dataset = load_dataset(data_dir, batch_size)

    print("Training...")
    for image, mask in train_dataset:
        print(image, mask)

    print("Testing...")
    for image, mask in test_dataset:
        print(image, mask)


if __name__ == "__main__":
    main()
