from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from typing_extensions import TypedDict, Required, NotRequired
else:
    from typing import TypedDict, Required, NotRequired


import numpy as np
from numpy.typing import ArrayLike

from warnings import warn

from tensorflow.keras import Model, saving
from tensorflow.keras.layers import (
    Layer,
    InputLayer,
    Reshape,
    Conv1D,
    BatchNormalization,
    MaxPool1D,
    Dropout,
    Permute,
    Flatten,
    Dense,
)

from .layers import ToHertzLayer
from .utils import PAD_TYPE, freq2cents, prepare_frames, prepare_signal, cents2freq


class LayerInfo(TypedDict):
    """Define a CNN layer

    Keys:
        filters (int): the dimension of the output space (the number of filters in the convolution).
        kernel_size (int): specifying the size of the convolution window.
        strides (int, optional): specifying the stride length of the convolution. Defaults to 1.
        maxpool_size (int, optional): size of the max pooling window. Defaults to 1 (no max pooling).
    """

    filters: Required[int]
    kernel_size: Required[int]
    strides: NotRequired[int]
    maxpool_size: NotRequired[int]


@saving.register_keras_serializable()
class CrepeModel(Model):
    """CREPE pitch estimation model

    Args:
        *layers (sequence of LayerInfo): Variable length argument list to define
            CNN layers.

        weights_file (str, optional): path to the weights file to load. It can
            either be a .weights.h5 file or a legacy .h5 weights file. Defaults
            to None.

        nb_input (int | None, optional): input frame size. Defaults to None to
            use the class default (1024).

        nb_freq_bins (int | None, optional): classifier output size. Defaults to
            None to use the class default (360).

        return_f0 (bool, optional): True to return pitch estimates in Hz.
            Defaults to False to return classifier output.

        framewise (bool | None, optional): True to transform the input to a
            sequence of sliding window frames. This option must be True or None
            for CrepeModel. Defaults to True.

        voice_threshold (float | None, optional): Classifier output threshold to
            detect voice. Defaults to None (uses the class default of 0.5.).

        dropout (float, optional): dropout rate (training only). Defaults to 0.25.
    """

    fs: int = 16000  # input sampling rate
    nb_input: int = 1024  #
    nb_freq_bins: int = 360  # number of frequency bins
    cmin: float = 1997.3794084376191  # minimum frequency bin in cents
    cmax: float = 7180 + 1997.3794084376191  # minimum frequency bin in cents
    fref: float = 10.0  # reference frequency for cents conversion
    tohz_nb_average: int = 9
    voice_threshold: float = 0.5

    def __init__(
        self,
        *layers: tuple[LayerInfo],
        weights_file: str | None = None,
        nb_input: int | None = None,
        nb_freq_bins: int | None = None,
        return_f0: bool = False,
        framewise: bool | None = True,
        voice_threshold: float | None = None,
        dropout: float = 0.25,
    ):
        if framewise is False:
            raise ValueError("CrepeModel only support framewise=True.")

        super().__init__()

        if nb_input is not None:
            self.nb_input = nb_input
        if nb_freq_bins is not None:
            self.nb_freq_bins = nb_freq_bins
        if voice_threshold is not None:
            self.voice_threshold = voice_threshold

        self.input_layers = [
            InputLayer(shape=(self.nb_input,), name="input", dtype="float32"),
            Reshape(target_shape=(self.nb_input, 1), name="input-reshape"),
        ]

        self.cnn_layers = [
            (
                Conv1D(
                    d["filters"],
                    d["kernel_size"],
                    strides=d["strides"] if "strides" in d else 1,
                    padding="same",
                    activation="relu",
                    name=f"conv{l+1}",
                ),
                BatchNormalization(name=f"conv{l+1}-BN"),
                MaxPool1D(
                    pool_size=d.get("maxpool_size", 1),
                    strides=None,
                    padding="valid",
                    name=f"conv{l+1}-maxpool_size",
                ),
                Dropout(dropout, name=f"conv{l+1}-dropout"),
            )
            for l, d in enumerate(layers)
        ]

        self.output_layers = [
            # Permute((2, 1, 3), name="transpose"),
            Flatten(name="flatten"),
            Dense(self.nb_freq_bins, activation="sigmoid", name="classifier"),
        ]

        self.return_f0 = return_f0
        if return_f0:
            self.output_layers.extend(
                [
                    Reshape(
                        target_shape=(1, self.nb_freq_bins),
                        name="to-hertz-reshape",
                    ),
                    ToHertzLayer(
                        threshold=self.voice_threshold,
                        cmin=self.cmin,
                        cmax=self.cmax,
                        fref=self.fref,
                        nb_average=self.tohz_nb_average,
                        name="to-hertz",
                    ),
                    Reshape(
                        target_shape=(2,),
                        name="output-reshape",
                    ),
                ]
            )

        self.build(None)

        if weights_file is not None:  # if restarting learning from a checkpoint
            self.load_weights(weights_file, by_name=True)

    @property
    def bin_frequencies(self) -> np.ndarray:
        """np.ndarray: frequencies of the classifier output bins"""
        return cents2freq(
            np.linspace(self.cmin, self.cmax, self.nb_freq_bins), self.fref
        )

    @property
    def layers(self) -> list[Layer]:
        """list[Layer]: All the model layers in the execution order"""
        return list(
            [
                *self.input_layers,
                *(l for ls in self.cnn_layers for l in ls),
                *self.output_layers,
            ]
        )

    def call(self, inputs, training=False):
        x = inputs
        for l in self.input_layers[1:]:
            x = l(x)
        for ccn, mp, bn, do in self.cnn_layers:
            x = ccn(x)
            x = mp(x)
            x = bn(x, training=training)
            x = do(x, training=training)
        for l in self.output_layers:
            x = l(x)

        return x

    def build(self, input_shape=None):

        # Build model
        layers = self.layers
        x = layers[0].output
        for layer in layers[1:]:
            layer.build(x.shape)
            x = layer(x)

        self.built = True

    def predict(
        self,
        x: ArrayLike,
        fs: int,
        hop: int | None = None,
        p0: int = 0,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates pitch predictions for the input signal.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `BatchNormalization` that behave differently during
        inference.

        Args:
            x: NumPy array (or array-like). Input signal(s). For a higher
                dimensional array, the pitch is detected along the last
                dimension.

            fs: Integer. Input signal sampling rate in Samples/second. This must
                match model.fs.

            hop: Integer. The increment in signal samples, by which the window
                is shifted in each step for frame-wise processing. If None, hop
                size of (roughly) 10 ms is used

            p0: Integer. The first element of the range of slices to calculate.
                If None then it is set to p_min, which is the smallest possible
                slice.

            p1: Integer or None. The end of the array. If None then the largest
                possible slice is used.

            k_offset: Integer. Index of first sample (t = 0) in x.

            padding: PAD_TYPE. Kind of values which are added, when the sliding window sticks out on
                either the lower or upper end of the input x. Zeros are added if the default ‘zeros’
                is set. For ‘edge’ either the first or the last value of x is used. ‘even’ pads by
                reflecting the signal on the first or last sample and ‘odd’ additionally multiplies
                it with -1.

            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).

            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.

            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.

            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                If self.return_f0 is true:
                    - t:  frame time stamps
                    - f0: predicted pitches
                    - confidence: pitch prediction confidences

                If self.return_f0 is false:
                    - f:  frequencies of classifier bins
                    - t:  frame time stamps
                    - P:  classifier output
        """

        if fs != self.fs:
            ValueError(
                f"fs({fs}) does not match the model sampling rate ({self.fs}). Resample x first."
            )

        m_num = self.nb_input

        if hop is None or hop <= 0:
            hop = int(10e-3 * fs)

        frames = prepare_frames(x, m_num, hop, p0, p1, k_offset, padding)
        inndim = frames.ndim
        inshape = frames.shape
        if inndim > 2:
            frames = frames.reshape(-1, m_num)

        out = super().predict(frames, **kwargs)
        # keywards: batch_size=None, verbose='auto', steps=None, callbacks=None

        if inndim > 1:
            out = out.reshape(*inshape[:-1], -1)

        t = (np.arange(-k_offset, out.shape[-2] - k_offset)) * (hop / fs)

        return (
            (t, *np.moveaxis(out, -1, 0))
            if self.return_f0
            else (self.bin_frequencies, t, out)
        )


###################################


@saving.register_keras_serializable()
class FcnF0Model(Model):

    fs: int = 8000  # input sampling rate
    nb_freq_bins: int = 486  # number of frequency bins
    classifier_kernel_size: int = 4
    cmin: float = freq2cents(30.0)  # minimum frequency bin in cents
    cmax: float = freq2cents(1000.0)  # minimum frequency bin in cents
    fref: float = 10.0  # reference frequency for cents conversion
    tohz_nb_average: int = 9
    voice_threshold: float = 0.5  #
    """FCN-F0 pitch estimation model

    Args:
        *layers (sequence of LayerInfo): Variable length argument list to define
            CNN layers.

        weights_file (str, optional): path to the weights file to load. It can
            either be a .weights.h5 file or a legacy .h5 weights file. Defaults
            to None.

        nb_freq_bins (int | None, optional): classifier output size. Defaults to
            None to use the class default (360).

        return_f0 (bool, optional): True to return pitch estimates in Hz.
            Defaults to False to return classifier output.

        framewise (bool | None, optional): True to transform the input to a
            sequence of sliding window frames. This option must be True or None
            for CrepeModel. Defaults to False.

        voice_threshold (float | None, optional): Classifier output threshold to
            detect voice. Defaults to None (uses the class default of 0.5.).

        dropout (float, optional): dropout rate (training only). Defaults to 0.25.
    """

    def __init__(
        self,
        *layers: tuple[LayerInfo],
        framewise: bool | None = False,
        weights_file=None,
        nb_freq_bins: int | None = None,
        classifier_kernel_size: int | None = None,
        return_f0: bool = False,
        voice_threshold: float | None = None,
        dropout: float = 0.25,
    ):
        super().__init__()

        if nb_freq_bins is not None:
            self.nb_freq_bins = nb_freq_bins
        if classifier_kernel_size is not None:
            self.classifier_kernel_size = classifier_kernel_size
        if voice_threshold is not None:
            self.voice_threshold = voice_threshold

        nb_input = self.classifier_kernel_size
        for d in reversed(layers):
            maxpool_size_size = d.get("maxpool_size", 1)
            nb_input = maxpool_size_size * nb_input + d["kernel_size"] - 1
        self.nb_input = nb_input
        self.framewise = bool(framewise)

        self.input_layers = (
            [
                InputLayer(shape=(nb_input,), name="input", dtype="float32"),
                Reshape(target_shape=(nb_input, 1), name="input-reshape"),
            ]
            if framewise
            else [
                InputLayer(shape=(None, 1), name="input", dtype="float32"),
                Reshape(target_shape=(-1, 1), name="input-reshape"),
            ]
        )

        self.cnn_layers = [
            (
                Conv1D(
                    d["filters"],
                    d["kernel_size"],
                    strides=d["strides"] if "strides" in d else 1,
                    padding="valid",
                    activation="relu",
                    name=f"conv{l+1}",
                ),
                (
                    MaxPool1D(
                        pool_size=d["maxpool_size"],
                        strides=None,
                        padding="valid",
                        name=f"conv{l+1}-maxpool_size",
                    )
                    if d.get("maxpool_size", 1) > 1
                    else None
                ),
                BatchNormalization(name=f"conv{l+1}-BN"),
                Dropout(dropout, name=f"conv{l+1}-dropout"),
            )
            for l, d in enumerate(layers)
        ]

        self.output_layers = [
            Conv1D(
                self.nb_freq_bins,
                self.classifier_kernel_size,
                strides=1,
                padding="valid",
                activation="sigmoid",
                name="classifier",
            )
        ]

        self.return_f0 = return_f0
        if return_f0:
            self.output_layers.append(
                ToHertzLayer(
                    threshold=self.voice_threshold,
                    cmin=self.cmin,
                    cmax=self.cmax,
                    fref=self.fref,
                    nb_average=self.tohz_nb_average,
                    name="to-hertz",
                )
            )

        if framewise:
            self.output_layers.append(
                Reshape(
                    target_shape=(2 if return_f0 else self.nb_freq_bins,),
                    name="output-reshape",
                ),
            )

        self.build(None)

        if weights_file is not None:  # if restarting learning from a checkpoint
            self.load_weights(weights_file, by_name=True)

    @property
    def layers(self) -> list[Layer]:
        """list[Layer]: All the model layers in the execution order"""
        return list(
            [
                *self.input_layers,
                *(l for ls in self.cnn_layers for l in ls if l is not None),
                *self.output_layers,
            ]
        )

    @property
    def native_hop(self) -> int:
        """int: hop size when operating non-framewise."""
        return np.prod([l.get_config().get("strides", [1])[0] for l in self.layers])

    @property
    def bin_frequencies(self) -> np.ndarray:
        """np.ndarray: frequencies of the classifier output bins"""
        return cents2freq(
            np.linspace(self.cmin, self.cmax, self.nb_freq_bins), self.fref
        )

    def call(self, inputs, training=False):

        x = inputs
        for l in self.input_layers[1:]:
            x = l(x)
        for ccn, mp, bn, do in self.cnn_layers:
            x = ccn(x)
            if mp is not None:
                x = mp(x)
            x = bn(x, training=training)
            x = do(x, training=training)

        for l in self.output_layers:
            x = l(x)

        if training:
            x = Permute((2, 1, 3), name="transpose", trainable=training)(x)
            x = Flatten(name="flatten", trainable=training)(x)

        return x

    def build(self, input_shape=None):

        # Build model
        layers = self.layers
        x = layers[0].output
        for layer in layers[1:]:
            layer.build(x.shape)
            x = layer(x)

        self.built = True

    def predict(
        self,
        x: ArrayLike,
        fs: int,
        hop: int | None = None,
        p0: int = 0,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates pitch predictions for the input signal.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `BatchNormalization` that behave differently during
        inference.

        Args:
            x: NumPy array (or array-like). Input signal(s). For a higher
                dimensional array, the pitch is detected along the last
                dimension.

            fs: Integer. Input signal sampling rate in Samples/second. This must
                match model.fs.

            hop: Integer. The increment in signal samples, by which the window
                is shifted in each step for frame-wise processing. If None, hop
                size of (roughly) 10 ms is used. For stream processing, this argument
                is ignored and the native hop size (self.native_hop) of the model
                is used instead.

            p0: Integer. The first element of the range of slices to calculate.
                If None then it is set to p_min, which is the smallest possible
                slice.

            p1: Integer or None. The end of the array. If None then the largest
                possible slice is used.

            k_offset: Integer. Index of first sample (t = 0) in x.

            padding: PAD_TYPE. Kind of values which are added, when the sliding window sticks out on
                either the lower or upper end of the input x. Zeros are added if the default ‘zeros’
                is set. For ‘edge’ either the first or the last value of x is used. ‘even’ pads by
                reflecting the signal on the first or last sample and ‘odd’ additionally multiplies
                it with -1.

            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).

            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.

            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.

            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                If self.return_f0 is true:
                    - t:  frame time stamps
                    - f0: predicted pitches
                    - confidence: pitch prediction confidences

                If self.return_f0 is false:
                    - f:  frequencies of classifier bins
                    - t:  frame time stamps
                    - P:  classifier output
        
        """

        if fs != self.fs:
            raise ValueError(
                f"fs({fs}) does not match the model sampling rate ({self.fs}). Resample x first."
            )

        m_num = self.nb_input

        if self.framewise:
            if hop is None or hop <= 0:
                hop = int(10e-3 * fs)

            frames = prepare_frames(x, m_num, hop, p0, p1, k_offset, padding)
            inndim = frames.ndim
            inshape = frames.shape
            if inndim > 2:
                frames = frames.reshape(-1, m_num)
        else:  # fully convolutional (frames determined by the model structure)
            if hop is not None and hop > 0:
                warn(
                    "For fully convolutional operation, model's native hop size is used, and hop argument is ignored."
                )
            hop = self.native_hop
            frames = prepare_signal(x, m_num, hop, p0, p1, k_offset, padding)
            inndim = frames.ndim
            inshape = frames.shape
            if inndim == 1:
                frames = frames.reshape(1, -1)
            elif inndim > 2:
                frames = frames.reshape(-1, inshape[-1])

        out = super().predict(frames, **kwargs)
        # keywards: batch_size=None, verbose='auto', steps=None, callbacks=None

        if self.framewise:
            if inndim > 1:
                out = out.reshape(*inshape[:-1], -1)
        else:
            if inndim == 1:
                out = out[0]
            elif inndim > 2:
                out = out.reshape(*inshape[:-1], *out.shape[-2:])

        t = (np.arange(-k_offset, out.shape[-2] - k_offset)) * (hop / fs)

        return (
            (t, *np.moveaxis(out, -1, 0))
            if self.return_f0
            else (self.bin_frequencies, t, out)
        )
