import argparse
import time
import datetime
from pathlib import Path

import tensorflow as tf

import segmentation_models as sm

from unet.data.dataset import make_dataset
from unet.models.modified_unet import modified_unet_model
from unet.visualization.plot import save_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="output", help="Output directory")
parser.add_argument(
    "--data_dir", default="data/processed", help="Directory containing input data"
)
parser.add_argument("--batch_size", default=2, help="Batch size")
parser.add_argument("--epochs", default=10, help="Number of epochs to train for")


def callbacks(output_dir: Path):
    # Save checkpoints.
    # checkpoint_path: Path = output_dir / "checkpoints" / "checkpoint.ckpt"
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path, save_weights_only=True,
    # )

    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir / "logs" / run_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks = []
    # callbacks.append(tensorboard_callback)
    return callbacks


def main():
    args = parser.parse_args()

    batch_size: int = int(args.batch_size)
    epochs: int = int(args.epochs)
    output_dir: Path = Path(args.output_dir).resolve()
    data_dir: Path = Path(args.data_dir).resolve()

    train_dataset, test_dataset = make_dataset(data_dir, batch_size)

    metric_iou = sm.metrics.IOUScore(threshold=0.5)
    model = modified_unet_model()
    model.compile(
        optimizer="Adam", loss=sm.losses.bce_dice_loss, metrics=[sm.metrics.iou_score]
    )

    t0: float = time.time()

    model.fit(train_dataset, epochs=epochs, callbacks=callbacks(output_dir))

    train_duration: float = time.time() - t0
    print(f"Training finished in {datetime.timedelta(seconds=train_duration)}")

    model.save(output_dir / "models" / "modified_unet")

    # Evaluation.

    result = model.evaluate(test_dataset)
    dict(zip(model.metrics_names, result))

    print(f"Saving predictions...")
    figure_path = output_dir / "figures"
    save_predictions(model, test_dataset, figure_path, 26)


if __name__ == "__main__":
    main()
