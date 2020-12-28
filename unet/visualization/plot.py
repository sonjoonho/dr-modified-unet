from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
from tqdm import tqdm

matplotlib.use("Agg")

MASK_THRESHOLD = 0.5


def threshold_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    # print(tf.math.reduce_max(pred_mask))
    return tf.cast(pred_mask > 0.0, tf.float32)
    # return pred_mask


def save_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset, output_dir: Path, nrows: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    titles: List[str] = ["Input Image", "True Mask", "Predicted Mask"]

    row: int = 0
    progress_bar: tqdm = tqdm(range(nrows))
    for image_batch, mask_batch in dataset:
        pred_masks: tf.Tensor = threshold_mask(model.predict(image_batch))
        for image, mask, pred_mask in zip(image_batch, mask_batch, pred_masks):
            fig, ax = plt.subplots(ncols=3)
            items = (image, mask, pred_mask)
            for k, item in enumerate(items):
                ax[k].axis("off")
                ax[k].imshow(
                    tf.keras.preprocessing.image.array_to_img(item), cmap="Greys_r"
                )

            output_path = output_dir / f"prediction_{row}.jpg"
            plt.savefig(str(output_path), bbox_inches="tight", dpi=200)
            plt.close(fig)

            row += 1
            progress_bar.update()
            if row >= nrows:
                progress_bar.close()
                return
