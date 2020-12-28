"""Functions to build a Tensorflow dataset from image data."""

from pathlib import Path
from typing import Tuple

import tensorflow as tf

from unet.data.files import filepaths, MaskPaths


def load_image(image_path: str, mask_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Obtains the image data from the path."""
    image_string: tf.Tensor = tf.io.read_file(image_path)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask_string: tf.Tensor = tf.io.read_file(mask_path)
    mask = tf.io.decode_jpeg(mask_string, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return image, mask


def perform_flips(image: tf.Tensor, flip_lr: bool, flip_ud: bool) -> tf.Tensor:
    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    return image


def train_preprocess(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Random numbers to determine if distortions are applied.
    distortions = tf.random.uniform([2], 0, 1.0)

    flip_lr = distortions[0] > 0.5
    flip_ud = distortions[1] > 0.5

    image = perform_flips(image, flip_lr, flip_ud)

    # We apply colour transformations to the image only.
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    mask = perform_flips(mask, flip_lr, flip_ud)
    mask = tf.cast(mask > 0.5, tf.float32)

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


def make_dataset(
    data_dir: Path, batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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

    train_length: int = len(train_image_paths)

    # Convert paths to strings.
    train_masks = [str(f) for f in train_mask_paths.od]
    test_masks = [str(f) for f in test_mask_paths.od]

    train_images = [str(f) for f in train_image_paths]
    test_images = [str(f) for f in test_image_paths]

    AUTOTUNE: int = tf.data.experimental.AUTOTUNE

    # Creating the training and test datasets.
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_ds = train_ds.shuffle(train_length, reshuffle_each_iteration=True)
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds
