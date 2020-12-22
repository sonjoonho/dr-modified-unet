from pathlib import Path
from typing import Tuple, List

import tensorflow as tf


def filenames(path: Path) -> List[str]:
    """Returns the filenames of all .jpg files in the specified path."""

    assert path.is_dir()

    # Sort to ensure that the images and masks match up correctly.
    return sorted([f.name for f in path.glob("*") if f.is_file()])


def class_filenames(
    masks_dir: Path,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Returns the filenames of masks in sub-directories."""

    assert masks_dir.is_dir()

    ma_dir: Path = masks_dir / "1. Microaneurysms"
    he_dir: Path = masks_dir / "2. Haemorrhages"
    ex_dir: Path = masks_dir / "3. Hard Exudates"
    se_dir: Path = masks_dir / "4. Soft Exudates"
    od_dir: Path = masks_dir / "5. Optic Disc"
    return (
        filenames(ma_dir),
        filenames(he_dir),
        filenames(ex_dir),
        filenames(se_dir),
        filenames(od_dir),
    )


def load_dataset(data_dir: Path, batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns the training and testing datasets."""

    assert data_dir.is_dir(), f"Directory {data_dir} does not exist"

    # Define the data directories
    train_images_dir: Path = data_dir / "1. Original Images" / "a. Training Set"
    train_masks_dir: Path = data_dir / "2. All Segmentation Groundtruths" / "a. Training Set"
    train_ma, train_he, train_ex, train_se, train_od = class_filenames(train_masks_dir)

    test_images_dir: Path = data_dir / "1. Original Images" / "b. Testing Set"
    test_masks_dir: Path = data_dir / "2. All Segmentation Groundtruths" / "b. Testing Set"
    test_ma, test_he, test_ex, test_se, test_od = class_filenames(test_masks_dir)

    # Retrieve filenames.
    train_images = filenames(train_images_dir)
    test_images = filenames(test_images_dir)

    # Creating the training and test datasets.
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_ma))
    train_ds = train_ds.shuffle(len(train_images))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(1)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_ma))

    return train_ds, test_ds
