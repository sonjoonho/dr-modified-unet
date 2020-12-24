"""Functions to build a Tensorflow dataset from image data."""

from pathlib import Path
from typing import Tuple

import tensorflow as tf

from unet.data.files import filepaths, MaskPaths


def load_image(image_path: str, mask_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Obtains the image data from the path."""
    image_string: tf.Tensor = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_string, channels=3, dtype=tf.float16)

    mask_string: tf.Tensor = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask_string, channels=1, dtype=tf.float16)

    image.set_shape([128, 128, 3])
    mask.set_shape([128, 128, 1])

    return image, mask


def mask_filepaths(masks_dir: Path) -> MaskPaths:
    """Returns the filenames of masks in sub-directories."""

    assert masks_dir.is_dir()

    ma_dir: Path = masks_dir / "ma"
    he_dir: Path = masks_dir / "he"
    ex_dir: Path = masks_dir / "ex"
    se_dir: Path = masks_dir / "se"
    od_dir: Path = masks_dir / "od"
    return MaskPaths(
        ma=filepaths(ma_dir),
        he=filepaths(he_dir),
        ex=filepaths(ex_dir),
        se=filepaths(se_dir),
        od=filepaths(od_dir),
    )


def make_dataset(data_dir: Path, batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns the training and testing datasets."""

    assert data_dir.is_dir(), f"Directory {data_dir} does not exist"

    # Define the data directories
    train_image_dir: Path = data_dir / "train" / "images"
    train_mask_dir: Path = data_dir / "train" / "masks"
    train_mask_paths = mask_filepaths(train_mask_dir)

    test_image_dir: Path = data_dir / "test" / "images"
    test_mask_dir: Path = data_dir / "test" / "masks"
    test_mask_paths = mask_filepaths(test_mask_dir)

    # Retrieve filenames.
    train_image_paths = filepaths(train_image_dir)
    test_image_paths = filepaths(test_image_dir)

    # Convert paths to strings.
    train_masks_od = [str(f) for f in train_mask_paths.od]
    test_masks_od = [str(f) for f in test_mask_paths.od]

    train_images = [str(f) for f in train_image_paths]
    test_images = [str(f) for f in test_image_paths]

    # Creating the training and test datasets.
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks_od))
    train_ds = train_ds.shuffle(len(train_image_paths))
    train_ds = train_ds.map(load_image)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(1)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks_od))
    test_ds = test_ds.map(load_image)

    return train_ds, test_ds
