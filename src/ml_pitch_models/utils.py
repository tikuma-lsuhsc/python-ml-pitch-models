from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike
from numpy.lib.stride_tricks import sliding_window_view


def freq2cents(f0: ArrayLike, f_ref: float = 1.0) -> ArrayLike:
    """Convert a given frequency into its corresponding cents value, according to given reference frequency f_ref

    Args:
        f0 (ArrayLike): f0 value (in Hz)
        f_ref (float, optional): reference frequency in Hz. Defaults to 1.0 Hz.

    Returns:
        ArrayLike: frequency in cents
    """

    c = 1200 * np.log2(f0 / f_ref)
    return c


def cents2freq(cents: ArrayLike, f_ref: float = 1.0) -> ArrayLike:
    """conversion from cents value to f0 in Hz

    Args:
        cents (ArrayLike): pitch frequency in cents
        f_ref (float, optional): reference frequency in Hz. Defaults to 1.0 Hz.

    Returns:
        ArrayLike: frequency in Hz
    """

    f0 = f_ref * 2 ** (cents / 1200)
    return f0


def sliding_norm(x: ArrayLike, m_num: int) -> ArrayLike:
    """Normalize each sample by mean and variance on a sliding window

    Args:
        x (np.ndarray): input signal. If multidimensional, the last dimension is the time axis
        m_num (int): size of the frames used during training, that should be used for the normalization

    Returns:
        ArrayLike: normalized input signal
    """

    m_num_mid = (m_num - 1) // 2
    pad_width = [(0, 0)] * (x.ndim - 1) + [(m_num_mid, m_num - m_num_mid - 1)]
    xpadded = np.pad(x, pad_width, mode="reflect", reflect_type="even")

    frames = sliding_window_view(xpadded, m_num, -1)

    # normalize each frame -- this is expected by the model
    mean = np.mean(frames, axis=-1)
    std = np.std(frames, axis=-1)
    std[std == 0.0] = 1
    # if constant (typically only zeros for audio signals), keep the same
    return (x - mean) / std


PAD_TYPE = Literal["zeros", "edge", "even", "odd"]
"""Kind of values which are added, when the sliding window sticks out on 
either the lower or upper end of the input x. 
    - 'zeros' - Zeros are added
    - 'edge'  - Either the first or the last value of x is used
    - 'even'  - Pads by reflecting the signal on the first or last sample
    - 'odd'   - Pads by reflecting the signal on the first or last sample 
                additionally multiplies it with -1.
"""


def get_pad_kws(padding: PAD_TYPE) -> dict:
    """convert padding option to np.pad arguments

    Args:
        padding (PAD_TYPE): Kind of values which are added, when the sliding window sticks out on either the lower or upper end of the input x.
            - 'zeros' - Zeros are added
            - 'edge'  - Either the first or the last value of x is used
            - 'even'  - Pads by reflecting the signal on the first or last sample
            - 'odd'   - Pads by reflecting the signal on the first or last sample
                        additionally multiplies it with -1.

    Returns:
        dict: associated np.pad keyword arguments
    """
    if padding not in (padding_types := get_args(PAD_TYPE)):
        raise ValueError(f"Parameter {padding=} not in {padding_types}!")
    return {  # possible keywords to pass to np.pad:
        "zeros": dict(mode="constant", constant_values=(0, 0)),
        "edge": dict(mode="edge"),
        "even": dict(mode="reflect", reflect_type="even"),
        "odd": dict(mode="reflect", reflect_type="odd"),
    }[padding]


def pad_signal(
    x: ArrayLike,
    m_num: int,
    hop: int,
    p0: int = 0,
    p1: int | None = None,
    k_off: int = 0,
    padding: PAD_TYPE = "zeros",
) -> np.ndarray:
    """pad (or trim) signal for short-time sliding window analysis

    The analysis windows of length m_num is sled over the last axis of the input
    signal x by hop increments.

    Let k0 = m_num//2 and k1 = m_num - k0. Then, the p-th window is spanning
    the signal indices: [p * hop - k0 + k_off, p*hop + k1 + k_off).

        Args:
            x (ArrayLike): signal array (time axis = last axis)
            m_num (int): frame size
            hop (int): increment in signal samples for sliding window
            p0 (int, optional): First window index. Defaults to 0.
            p1 (int | None, optional): End of interval (last window is p1-1).
                Defaults to None to be the latest window with its center within
                the signal duration.
            k_off (int, optional): Index of first sample (t = 0) in `x`. Defaults to 0.
            padding (PAD_TYPE, optional): signal padding mode. Defaults to "zeros".

        Returns:
            np.ndarray: padded/trimmed signal to cover windows p0 to p1

    """

    m_num_mid = m_num // 2  # frame center point
    n = x.shape[-1]

    if p0 is None:
        p0 = -k_off // hop

    if p1 is None:
        p1 = (n - k_off) // hop

    n1 = (p1 - p0) * hop
    i0 = p0 * hop - m_num_mid + k_off  # start sample
    i1 = i0 + n1 + m_num  # end sample

    pad_kws = get_pad_kws(padding)

    # dimensions for padding x:
    pad_width = [(0, 0)] * (x.ndim - 1) + [(-min(i0, 0), max(i1 - n, 0))]

    return np.pad(x[..., max(i0, 0) : min(i1, n)], pad_width, **pad_kws)


def prepare_signal(
    x: ArrayLike,
    m_num: int,
    hop: int,
    p0: int | None = None,
    p1: int | None = None,
    k_off: int = 0,
    padding: PAD_TYPE = "zeros",
) -> np.ndarray:
    """normalize and pad signal

    Args:
        x (ArrayLike): signal array (time axis = last axis)
        m_num (int): frame size
        hop (int): increment in signal samples for sliding window
        p0 (int, optional): First window index. Defaults to 0.
        p1 (int | None, optional): End of interval (last window is p1-1).
            Defaults to None to be the latest window with its center within
            the signal duration.
        k_off (int, optional): Index of first sample (t = 0) in `x`. Defaults to 0.
        padding (PAD_TYPE, optional): signal padding mode. Defaults to "zeros".

    Returns:
        np.ndarray: normalized then padded/trimmed signal to cover windows p0 to p1
    """

    # normalize the signal samples
    x = sliding_norm(x, m_num)

    return pad_signal(x, m_num, hop, p0, p1, k_off, padding)


def prepare_frames(
    x: ArrayLike,
    m_num: int,
    hop: int,
    p0: int | None = None,
    p1: int | None = None,
    k_off: int = 0,
    padding: PAD_TYPE = "zeros",
) -> np.ndarray:
    """Create a normalized sliding window view into the padded signal.

    Args:
        x (ArrayLike): signal array (time axis = last axis)
        m_num (int): frame size
        hop (int): increment in signal samples for sliding window
        p0 (int, optional): First window index. Defaults to 0.
        p1 (int | None, optional): End of interval (last window is p1-1).
            Defaults to None to be the latest window with its center within
            the signal duration.
        k_off (int, optional): Index of first sample (t = 0) in `x`. Defaults to 0.
        padding (PAD_TYPE, optional): signal padding mode. Defaults to "zeros".

    Returns:
        np.ndarray: array view containing windows from p0 to p1. The frame axis
        is the second from the last dimension.
    """

    frames = sliding_window_view(
        pad_signal(x, m_num, hop, p0, p1, k_off, padding), m_num, -1
    )[..., ::hop, :]

    # normalize each frame -- this is expected by the model
    std = np.std(frames, axis=-1, keepdims=True)
    std[std == 0.0] = 1
    return (frames - np.mean(frames, axis=-1, keepdims=True)) / std
