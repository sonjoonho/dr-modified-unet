import tensorflow as tf

import segmentation_models as sm


def modified_unet_model() -> tf.keras.Model:
    return sm.Unet("resnet34", encoder_weights="imagenet", activation="relu")
