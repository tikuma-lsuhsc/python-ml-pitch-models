from __future__ import annotations

from typing import Literal, get_args
from numpy.typing import ArrayLike

import sysconfig
from os import path
from glob import glob
from functools import cache

import numpy as np

from .models import FcnF0Model, CrepeModel, LayerInfo

PretrainedModelName = Literal[
    "crepe_full",
    "crepe_large",
    "crepe_medium",
    "crepe_small",
    "crepe_tiny",
    "fcn_1953",
    "fcn_929",
    "fcn_993",
]  # names of pretrained pitch detector models


@cache
def get_weights_path(model: PretrainedModelName) -> str:
    """pretrained weight file path

    Args:
        model (PretrainedModelName): pitch detector model name.

    Returns:
        str: path to an alternate weights file to load.
    """
    weights_path = path.join(
        sysconfig.get_path("purelib"),
        "ml_pitch_models",
        "data",
        model,
        "weights.h5",
    )

    if not path.exists(weights_path):
        raise FileNotFoundError(
            f"'{model}' weights has not been installed. Run pip install ml_pitch_models-data-{model.replace('_','-')}"
            if model in get_args(PretrainedModelName)
            else f"'{model}' is not a valid pre-trained model name. Run ml_pitch_models.available_models() to get a list of installed models."
        )

    return weights_path


@cache
def available_models() -> list[str]:
    """get a list of pretrained models available

    Returns:
        list[str]: list of available models
    """
    root_dir = path.join(sysconfig.get_path("purelib"), "ml_pitch_models", "data")
    files = glob(
        path.join("*", "weights.h5"),
        root_dir=root_dir,
    )
    return sorted(path.dirname(f) for f in files)


crepe_base_cnn_config = (
    {"filters": 32, "kernel_size": 512, "maxpool_size": 2, "strides": 4},
    {"filters": 4, "kernel_size": 64, "maxpool_size": 2},
    {"filters": 4, "kernel_size": 64, "maxpool_size": 2},
    {"filters": 4, "kernel_size": 64, "maxpool_size": 2},
    {"filters": 8, "kernel_size": 64, "maxpool_size": 2},
    {"filters": 16, "kernel_size": 64, "maxpool_size": 2},
)


@cache
def crepe_cnn_config(multiplier: int) -> tuple[LayerInfo]:
    """return CREPE model CNN layer information

    Args:
        multiplier (int): multiplier for the number of filters with respect to
        crepe_base_cnn_config.

    Returns:
        tuple[LayerInfo]: CNN layer information
    """
    return tuple(
        {k: v if k != "filters" else multiplier * v for k, v in d.items()}
        for d in crepe_base_cnn_config
    )


class CrepeFullModel(CrepeModel):
    """Full-size CREPE pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = crepe_cnn_config(32)

    def __init__(self, weights_file: str = None, **kwargs):
        if weights_file is None:
            weights_file = get_weights_path("crepe_full")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)


class CrepeLargeModel(CrepeModel):
    """Large-size CREPE pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = crepe_cnn_config(24)

    def __init__(self, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("crepe_large")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)


class CrepeMediumModel(CrepeModel):
    """Medium-size CREPE pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = crepe_cnn_config(16)

    def __init__(self, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("crepe_medium")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)


class CrepeSmallModel(CrepeModel):
    """Small-size CREPE pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = crepe_cnn_config(8)

    def __init__(self, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("crepe_small")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)


class CrepeTinyModel(CrepeModel):
    """Tiny-size CREPE pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = crepe_cnn_config(4)

    def __init__(self, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("crepe_tiny")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)


class FCN1953Model(FcnF0Model):
    """FCN-F0 FCN-1953 pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = (
        {"filters": 256, "kernel_size": 32, "maxpool": 2},
        {"filters": 32, "kernel_size": 64, "maxpool": 2},
        {"filters": 32, "kernel_size": 64, "maxpool": 2},
        {"filters": 128, "kernel_size": 64},
        {"filters": 256, "kernel_size": 64},
        {"filters": 512, "kernel_size": 64},
    )

    def __init__(self, *, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("fcn_1953")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)
        self.weights_file = weights_file


class FCN993Model(FcnF0Model):
    """FCN-F0 FCN-993 pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = (
        {"filters": 256, "kernel_size": 32, "maxpool": 2},
        {"filters": 32, "kernel_size": 32, "maxpool": 2},
        {"filters": 32, "kernel_size": 32, "maxpool": 2},
        {"filters": 128, "kernel_size": 32},
        {"filters": 256, "kernel_size": 32},
        {"filters": 512, "kernel_size": 32},
    )

    def __init__(self, *, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("fcn_993")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)
        self.weights_file = weights_file


class FCN929Model(FcnF0Model):
    """FCN-F0 FCN-929 pitch detector model

    Args:
        weights_file (str, optional): path to an alternate weights file to load.
            It can either be a .weights.h5 file or a legacy .h5 weights file.
            Defaults to None to load the pretrained weights.
        **kwargs: other CrepeModel constructor keyword arguments.
    """

    cnn_config = (
        {"filters": 256, "kernel_size": 32, "maxpool": 2},
        {"filters": 32, "kernel_size": 64, "maxpool": 2},
        {"filters": 128, "kernel_size": 64},
        {"filters": 256, "kernel_size": 64},
        {"filters": 512, "kernel_size": 64},
    )

    def __init__(self, *, weights_file: str = None, **kwargs):

        if weights_file is None:
            weights_file = get_weights_path("fcn_929")
        super().__init__(*self.cnn_config, weights_file=weights_file, **kwargs)
        self.weights_file = weights_file


def predict(
    x: ArrayLike,
    fs: int,
    model: PretrainedModelName | FcnF0Model | CrepeModel = "fcn_993",
    framewise: bool | None = None,
    voice_threshold: float = 0.5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates pitch predictions for the input signal.

    Args:
        x: Input signal(s). For a higher dimensional array, the pitch is detected along the last
            dimension.

        fs: Input signal sampling rate in Samples/second. This must
            match model.fs.

        model: Pitch detection deep-learning model.

        framewise: True to perform per-frame operation. If not given, CREPE models set this True
            while FCN-F0 models set False.

        voice_threshold: Voice detection threshold on the classifier confidence level. If unvoiced
            is detected, f0=0 is returned and its confidence level indicates the confidence of
            detecting unvoiced (i.e., 1 - (classifier confidence)).

        hop: The increment in signal samples, by which the window
            is shifted in each step for frame-wise processing. If None, hop
            size of (roughly) 10 ms is used. For stream processing, this argument
            is ignored and the native hop size (self.native_hop) of the model
            is used instead.

        p0: The first element of the range of slices to calculate.
            If None then it is set to p_min, which is the smallest possible
            slice.

        p1: The end of the array. If None then the largest
            possible slice is used.

        k_offset: Index of first sample (t = 0) in x.

        padding: Kind of values which are added, when the sliding window sticks out on
            either the lower or upper end of the input x. Zeros are added if the default ‘zeros’
            is set. For ‘edge’ either the first or the last value of x is used. ‘even’ pads by
            reflecting the signal on the first or last sample and ‘odd’ additionally multiplies
            it with -1.

        batch_size: Number of samples per batch.
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
            - t:  frame time stamps
            - f0: predicted pitches
            - confidence: pitch prediction confidences

    """
    if isinstance(model, str):
        try:
            model = {
                "crepe_full": CrepeFullModel,
                "crepe_large": CrepeLargeModel,
                "crepe_medium": CrepeMediumModel,
                "crepe_small": CrepeSmallModel,
                "crepe_tiny": CrepeTinyModel,
                "fcn_1953": FCN1953Model,
                "fcn_929": FCN929Model,
                "fcn_993": FCN993Model,
            }[model](
                framewise=framewise, return_f0=True, voice_threshold=voice_threshold
            )
        except IndexError:
            raise ValueError("Invalid model name")
        except ValueError:
            raise ValueError("CREPE model must use framewise=True")
    elif not isinstance(model, (CrepeFullModel, FcnF0Model)):
        raise ValueError(
            "model must be a valid pretained model name or a CREPE or FCN-F0 model instances."
        )

    return model.predict(x, fs, **kwargs)
