from __future__ import annotations

from collections.abc import Callable
import sys

if sys.version_info < (3, 11):
    from typing_extensions import TypedDict, Required, NotRequired, Literal
else:
    from typing import TypedDict, Required, NotRequired, Literal
from numpy.typing import NDArray

from functools import cached_property, partial
import numpy as np
from numpy.typing import ArrayLike
from numbers import Number

from warnings import warn

from tensorflow.keras import saving, Model
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

from .utils import PAD_TYPE, freq2cents, viterbi_cat, viterbi_conf, tohertz
from .stap import ShortTimeProcess, ShortTimeStreamProcess
from .layers import ToHertzLayer


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


class BaseMLPitchModel(Model):

    _harmonic_threshold: float = 0.5
    _tohz_nb_average: int = 9
    _postproc: Callable | None = None
    _postproc_detects_nh: bool = False

    def __init__(
        self,
        harmonic_threshold: float | None,
        postprocessor: Literal["viterbi", "hmm"] | None,
        postprocessor_kws: dict | None,
    ):
        """base class for machine-learning pitch detector models

        Parameters
        ----------
        harmonic_threshold
            Specify a confidence level threshold to separate harmonic and nonharmonic frames.
        postprocessor
            `'viterbi'` to use the viterbi algorithm to resolve the path of the highest collective confidence,
            or `'hmm'` to use the categorical hidden Markov model to resolve the path. None to skip post-processing.
        postprocessor_kws
            keyword arguments for the arguments to run`'viterbi'` or `'hmm'` postprocessor. If None, the default
            configurations is used.
        """
        super().__init__()

        if harmonic_threshold is not None:
            if not (isinstance(harmonic_threshold, Number) and harmonic_threshold >= 0):
                raise ValueError(f"{harmonic_threshold=} must be a nonnegative number.")
            self._harmonic_threshold = harmonic_threshold
        if postprocessor == "viterbi":
            self._postproc = partial(viterbi_conf, **(postprocessor_kws or {}))
            self._postproc_detects_nh = True
        elif postprocessor == "hmm":
            self._postproc = partial(viterbi_cat, **(postprocessor_kws or {}))
        elif postprocessor is not None:
            raise ValueError(f"{postprocessor=} is not a valid option.")

    @property
    def harmonic_threshold(self) -> float:
        """Voice detection threshold"""
        return self._harmonic_threshold

    @property
    def tohz_nb_average(self) -> float:
        """Number of samples calculate the weighted-average estimate of the pitch"""
        return self._tohz_nb_average

    def _create_to_pitch_layers(self, name: str = "to-hertz") -> Layer:
        return ToHertzLayer(
            fbins=self.f,
            threshold=self.harmonic_threshold,
            nb_average=self.tohz_nb_average,
            name=name,
        )

    def _tohertz(self, conf: NDArray, center: NDArray = None):
        return tohertz(conf, self.f, center, self.harmonic_threshold, self._postproc_detects_nh)


@saving.register_keras_serializable()
class CrepeModel(BaseMLPitchModel, ShortTimeProcess):
    """CREPE pitch estimation model

    Args:
        *layers
            Variable length argument list to define CNN layers.

        weights_file
            path to the weights file to load. It can
            either be a .weights.h5 file or a legacy .h5 weights file. Defaults
            to None.

        hop
            The increment in signal samples, by which the window
            is shifted in each step for frame-wise processing. If None, hop
            size of (roughly) 10 ms is used

        return_f0
            True to return pitch estimates in Hz.
            Defaults to False to return classifier output.

        framewise
            True to transform the input to a
            sequence of sliding window frames. This option must be True or None
            for CrepeModel. Defaults to True.

        harmonic_threshold
            Classifier output threshold to
            detect voice. Defaults to None (uses the class default of 0.5.).

        dropout
            Dropout rate (training only). Defaults to 0.25.

    """

    fs: int = 16000  # input sampling rate
    nb_input: int = 1024  #
    nb_freq_bins: int = 360  # number of frequency bins
    f_scale: str = "log"
    f_min: float = freq2cents(31.7)  # minimum frequency bin in cents
    delta_f: float = 20.0
    # cmax: float = 7180 + 1997.3794084376191  # minimum frequency bin in cents

    def __init__(
        self,
        *layers: tuple[LayerInfo],
        weights_file: str | None = None,
        hop: int | None = None,
        return_f0: bool = False,
        harmonic_threshold: float | None = None,
        postprocessor: Literal["viterbi", "hmm"] | None = None,
        postprocessor_kws: dict | None = None,
        dropout: float = 0.25,
    ):
        if hop is None:
            hop = int(10e-3 * CrepeModel.fs)

        ShortTimeProcess.__init__(self, hop)
        BaseMLPitchModel.__init__(
            self, harmonic_threshold, postprocessor, postprocessor_kws
        )

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
        if return_f0 and self._postproc is None:
            self.output_layers.extend(
                [
                    Reshape(
                        target_shape=(1, self.nb_freq_bins), name="to-hertz-reshape"
                    ),
                    self._create_to_pitch_layers("to-hertz"),
                    Reshape(target_shape=(2,), name="output-reshape"),
                ]
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
        p0: int = 0,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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
            x
                Input signal(s). For a higher
                dimensional array, the pitch is detected along the last
                dimension.

            fs
                Input signal sampling rate in Samples/second. This must
                match model.fs.

            p0
                The first element of the range of slices to calculate.
                If None then it is set to p_min, which is the smallest possible
                slice.

            p1
                The end of the array. If None then the largest
                possible slice is used.

            k_offset
                Index of first sample (t = 0) in x.

            padding
                Kind of values which are added, when the sliding window sticks out on
                either the lower or upper end of the input x. Zeros are added if the default ‘zeros’
                is set. For ‘edge’ either the first or the last value of x is used. ‘even’ pads by
                reflecting the signal on the first or last sample and ‘odd’ additionally multiplies
                it with -1.

            axis
                The axis of `x` over which to run the model along.
                If not given, the last axis is used.

            batch_size
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).

            verbose
                `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.

            steps
                Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.

            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        :Returns:
            If postprocessor is assigned:
                - f0: predicted pitches
                
            If self.return_f0 is true:
                - f0: predicted pitches
                - confidence: pitch prediction confidences

            If self.return_f0 is false:
                - 2D array of a sequence of confidence levels of all frequency bins
        """

        if fs != self.fs:
            ValueError(
                f"fs({fs}) does not match the model sampling rate ({self.fs}). Resample x first."
            )

        frames = self.prepare_frames(x, p0, p1, k_offset, padding, axis)
        inndim = frames.ndim
        inshape = frames.shape
        if inndim > 2:
            frames = frames.reshape(-1, self.nb_input)

        out = super().predict(frames, **kwargs)
        # keywards: batch_size=None, verbose='auto', steps=None, callbacks=None

        if inndim > 1:
            out = out.reshape(*inshape[:-1], -1)

        if self._postproc is None:
            return (*np.moveaxis(out, -1, 0),) if self.return_f0 else out
        else:
            p = self._postproc(out, self.f)
            return self._tohertz(out, p)


###################################


@saving.register_keras_serializable()
class FcnF0Model(BaseMLPitchModel, ShortTimeStreamProcess):

    fs: int = 8000  # input sampling rate
    nb_freq_bins: int = 486  # number of frequency bins
    f_scale: str = "log"
    f_min: float = freq2cents(30.0)  # minimum frequency bin in Hz
    delta_f: float = freq2cents(1000 / 30) / 485  # frequency spacing in cents

    classifier_kernel_size: int = 4

    _nb_input: int

    """FCN-F0 pitch estimation model

    Args:
        *layers (a sequence of LayerInfo): Variable length argument list to define
            CNN layers.

        weights_file: path to the weights file to load. It can
            either be a .weights.h5 file or a legacy .h5 weights file. Defaults
            to None.

        hop
            The increment in signal samples, by which the window
            is shifted in each step for frame-wise processing. If None or 0, the 
            model operates in fully convolutional (single-batch) mode with its 
            hop size set to the `native_hop`.

        return_f0: True to return pitch estimates in Hz.
            Defaults to False to return classifier output.

        harmonic_threshold
            Classifier output threshold to
            detect voice. Defaults to None (uses the class default of 0.5.).

        dropout
            Dropout rate (training only). Defaults to 0.25.
    """

    def __init__(
        self,
        *layers: tuple[LayerInfo],
        weights_file: str | None = None,
        hop: int | None | Literal["native"] = "native",
        return_f0: bool = False,
        harmonic_threshold: float | None = None,
        postprocessor: Literal["viterbi", "hmm"] | None = None,
        postprocessor_kws: dict | None = None,
        dropout: float = 0.25,
    ):
        if hop is None or hop == "native":
            hop = 0

        ShortTimeStreamProcess.__init__(self, hop)
        BaseMLPitchModel.__init__(
            self, harmonic_threshold, postprocessor, postprocessor_kws
        )

        framewise = bool(self._hop)

        nb_input = self.classifier_kernel_size
        for d in reversed(layers):
            maxpool_size_size = d.get("maxpool_size", 1)
            nb_input = maxpool_size_size * nb_input + d["kernel_size"] - 1
        self._nb_input = nb_input

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
        if return_f0 and self._postproc is None:
            self.output_layers.append(self._create_to_pitch_layers("to-hertz"))

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
    def nb_input(self) -> int:
        """Number of inputs to the model (readonly)"""
        return self._nb_input

    @cached_property
    def native_hop(self) -> int:
        """hop size when streaming (readonly)"""
        return np.prod([l.get_config().get("strides", [1])[0] for l in self.layers])

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
        p0: int | None = None,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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
            x
                Input signal(s). For a higher
                dimensional array, the pitch is detected along the last
                dimension.

            fs
                Input signal sampling rate in Samples/second. This must
                match model.fs.

            p0
                The first element of the range of slices to calculate.
                If None then it is set to p_min, which is the smallest possible
                slice.

            p1
                The end of the array. If None then the largest
                possible slice is used.

            k_offset
                Index of first sample (t = 0) in x.

            padding
                Kind of values which are added, when the sliding window sticks out on
                either the lower or upper end of the input x. Zeros are added if the default ‘zeros’
                is set. For ‘edge’ either the first or the last value of x is used. ‘even’ pads by
                reflecting the signal on the first or last sample and ‘odd’ additionally multiplies
                it with -1.

            axis
                The axis of `x` over which to run the model along.
                If not given, the last axis is used.

            batch_size
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).

            verbose
                `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.

            steps
                Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.

            callbacks
                List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        :Returns:
            If postprocessor is assigned:
                - f0: predicted pitches

            If self.return_f0 is true:
                - f0: predicted pitches
                - confidence: pitch prediction confidences

            If self.return_f0 is false:
                - 2D array of a sequence of confidence levels of all frequency bins

        """

        if fs != self.fs:
            raise ValueError(
                f"fs({fs}) does not match the model sampling rate ({self.fs}). Resample x first."
            )

        m_num = self.nb_input

        frames = self.prepare_frames(x, p0, p1, k_offset, padding, axis)
        inndim = frames.ndim
        inshape = frames.shape
        framewise = bool(self._hop)
        if framewise:  # batch operation
            if inndim > 2:
                frames = frames.reshape(-1, m_num)
        else:  # fully convolutional (frames determined by the model structure)
            if inndim == 1:
                frames = frames.reshape(1, -1)
            elif inndim > 2:
                frames = frames.reshape(-1, inshape[-1])

        out = super().predict(frames, **kwargs)
        # keywards: batch_size=None, verbose='auto', steps=None, callbacks=None

        if framewise:
            if inndim > 1:
                out = out.reshape(*inshape[:-1], -1)
        else:
            if inndim == 1:
                out = out[0]
            elif inndim > 2:
                out = out.reshape(*inshape[:-1], *out.shape[-2:])

        if self._postproc is None:
            return (*np.moveaxis(out, -1, 0),) if self.return_f0 else out
        else:
            p = self._postproc(out, self.f)
            return self._tohertz(out, p)
