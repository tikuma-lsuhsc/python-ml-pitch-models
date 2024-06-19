from __future__ import annotations

from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow import math as tfmath
import tensorflow as tf


class ToHertzLayer(Layer):
    """Classifer output to Hertz conversion layer

    Args:
        threshold (float): voicing threshold (0,1)
        cmin (float): minimum ouput frequency in cents
        cmax (float): maximum ouput frequency in cents
        nb_average (int, optional): _description_. Defaults to 9.
        fref (float, optional): _description_. Defaults to 10.0.
        name (str | None, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        fbins: np.ndarray,
        threshold: float,
        nb_average: int = 9,
        name: str | None = None,
    ):
        super().__init__(trainable=False, name=name)

        self.threshold = threshold

        # the bin number-to-cents bin_freqs
        self.bin_freqs = tf.reshape(tf.constant(fbins, tf.dtypes.float32), (1, 1, -1))
        self.index_delta = tf.reshape(tf.range(nb_average), (1, 1, -1))
        self.offset = nb_average // 2

        self.bin_freqs: tf.Tensor
        self.start_max: tf.dtypes.int32
        self.built: bool

    def get_config(self):
        config = super().get_config()
        config.update({"bin_freqs": self.bin_freqs})
        return config

    def build(self, input_shape):
        ndim = len(input_shape)
        if ndim != 3:
            raise ValueError(
                "ToHertzLayer expects its input to be 3D Tensor (batch_shape, steps, activation_levels)"
            )
        if input_shape[-1] != self.bin_freqs.shape[-1]:
            raise ValueError(
                f"ToHertzLayer the number of activation_levels to be {self.bin_freqs.shape[-1]}"
            )

        self.start_max = input_shape[-1] - self.index_delta.shape[-1]
        self.built = True

    def call(self, inputs):

        # peak index
        center = tfmath.argmax(inputs, axis=-1, output_type=tf.dtypes.int32)
        shape = tf.shape(inputs)
        start = tf.reshape(
            tfmath.minimum(self.start_max, tfmath.maximum(0, center - self.offset)),
            [shape[0], shape[1], 1],
        )
        indices = start + self.index_delta

        # weighted mean of 9 values
        weights = tf.experimental.numpy.take_along_axis(inputs, indices, axis=2)
        c = tf.experimental.numpy.take_along_axis(self.bin_freqs, indices, axis=2)

        product_sum = tfmath.reduce_sum(c * weights, axis=2)
        weight_sum = tfmath.reduce_sum(weights, axis=2)
        f = product_sum / weight_sum

        confidence = tfmath.reduce_max(inputs, axis=-1)  # keepdims?

        # voice detector
        voiced = confidence > self.threshold
        f = tf.where(voiced, f, 0.0)
        confidence = tf.where(voiced, confidence, 1.0 - confidence)

        return tf.stack([f, confidence], axis=2)
