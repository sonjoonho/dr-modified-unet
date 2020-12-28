"""Functions to process data.

```
processed/
    train/
    test/
```
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from data import image
from data.files import filepaths, MaskPaths

IMAGE_SIZE = 512

parser = argparse.ArgumentParser()
parser.add_argument(
    "--idrid_dir", default="../data/raw/IDRiD", help="Directory with the IDRiD dataset"
)
parser.add_argument(
    "--output_dir",
    default="../data/processed",
    help="Directory to write processed data to",
)


def transform_and_save(img: Image.Image, output_path: Path) -> Image.Image:
    img = image.pad(img)
    img = image.center_crop(img, 3600)
    img = image.resize(img, IMAGE_SIZE)

    img.save(output_path, "JPEG", quality=100, subsampling=0)
    return img


def _preprocess_class_masks(paths: List[Path], output_dir: Path, progress_bar: tqdm):
    """Helper for `convert_masks`."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in paths:
        output_filename = input_path.stem + ".jpg"
        output_path = output_dir / output_filename

        assert input_path.suffix in (".tif", ".tiff"), "Input file is not a TIFF file"

        img: Image.Image = Image.open(input_path)

        img = image.binarise(img)
        transform_and_save(img, output_path)
        img.close()

        progress_bar.update()


def preprocess_masks(paths: MaskPaths, output_dir: Path):
    """Converts masks from .tif to .jpg. Creates output_dir if it does not exist."""

    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(range(paths.total()), file=sys.stdout)

    _preprocess_class_masks(paths.ma, output_dir / "ma", progress_bar)
    _preprocess_class_masks(paths.he, output_dir / "he", progress_bar)
    _preprocess_class_masks(paths.ex, output_dir / "ex", progress_bar)
    _preprocess_class_masks(paths.se, output_dir / "se", progress_bar)
    _preprocess_class_masks(paths.od, output_dir / "od", progress_bar)

    progress_bar.close()


def preprocess_images(train_image_paths: List[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(train_image_paths, file=sys.stdout):
        output_filename = input_path.stem + ".jpg"
        output_path = output_dir / output_filename

        img: Image.Image = Image.open(input_path)
        transform_and_save(img, output_path)
        img.close()


def raw_mask_filepaths(masks_dir: Path,) -> MaskPaths:
    """Returns the filenames of masks in sub-directories."""

    assert masks_dir.is_dir()

    ma_dir: Path = masks_dir / "1. Microaneurysms"
    he_dir: Path = masks_dir / "2. Haemorrhages"
    ex_dir: Path = masks_dir / "3. Hard Exudates"
    se_dir: Path = masks_dir / "4. Soft Exudates"
    od_dir: Path = masks_dir / "5. Optic Disc"

    return MaskPaths(
        ma=filepaths(ma_dir),
        he=filepaths(he_dir),
        ex=filepaths(ex_dir),
        se=filepaths(se_dir),
        od=filepaths(od_dir),
    )


def main():
    args = parser.parse_args()

    idrid_dir: Path = Path(args.idrid_dir).resolve()
    output_dir: Path = Path(args.output_dir).resolve()

    assert idrid_dir.is_dir(), f"Directory {idrid_dir} does not exist"

    # Define the data directories
    train_image_dir: Path = idrid_dir / "1. Original Images" / "a. Training Set"
    train_mask_dir: Path = idrid_dir / "2. All Segmentation Groundtruths" / "a. Training Set"
    train_mask_paths = raw_mask_filepaths(train_mask_dir)

    test_image_dir: Path = idrid_dir / "1. Original Images" / "b. Testing Set"
    test_mask_dir: Path = idrid_dir / "2. All Segmentation Groundtruths" / "b. Testing Set"
    test_mask_paths: MaskPaths = raw_mask_filepaths(test_mask_dir)

    # Retrieve filenames.
    train_image_paths: List[Path] = filepaths(train_image_dir)
    test_image_paths: List[Path] = filepaths(test_image_dir)

    print("Preprocessing training images...")
    preprocess_images(train_image_paths, output_dir / "train" / "images")

    print("Preprocessing training masks...")
    preprocess_masks(train_mask_paths, output_dir / "train" / "masks")

    print("Preprocessing testing images...")
    preprocess_images(test_image_paths, output_dir / "test" / "images")

    print("Preprocessing testing masks...")
    preprocess_masks(test_mask_paths, output_dir / "test" / "masks")


if __name__ == "__main__":
    main()
