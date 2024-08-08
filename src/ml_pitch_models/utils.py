from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
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
) -> NDArray:
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
) -> NDArray:
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
        NDArray: normalized then padded/trimmed signal to cover windows p0 to p1
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
) -> NDArray:
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


def viterbi_cat(
    salience: NDArray,
    f: NDArray,
    smoothing_factor: int = 12,
    self_emission: float = 0.1,
) -> NDArray:
    """Find the pitch transition path using the categorical hidden Markov model to induce pitch
    continuity.

    Parameters
    ----------
    salience
        2D matrix (number of frames by number of pitch candidates) of deep-learning model outcome
    f
        1D vector of pitch candidate frequencies
    smoothing_factor, optional
        maximum allowable pitch transition in frequency bins, by default 12
    self_emission, optional
        self-emission probability, by default 0.1

    Returns
    -------
        The chosen sequence of frequency bin indices
    """
    from hmmlearn import hmm

    vecSize = salience.shape[1]

    # uniform prior on the starting pitch
    starting = np.ones(vecSize) / vecSize

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(vecSize), range(vecSize))
    transition = np.maximum(smoothing_factor - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    emission = np.eye(vecSize) * self_emission + np.ones(shape=(vecSize, vecSize)) * (
        (1 - self_emission) / vecSize
    )

    # fix the model parameters because we are not optimizing the model
    model = hmm.CategoricalHMM(vecSize, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = (
        starting,
        transition,
        emission,
    )

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return path


def viterbi_conf(
    conf: NDArray,
    f: NDArray,
    eta_s: float = 0.25,
    eta_v: float = 0.75,
    rho_o: float = 0.01,
    rho_oj: float = 0.35,
    vu_high_cost: float = 0.5,
    vu_low_cost: float = 0.05,
) -> list[int]:
    """find the pitch transition path based on the pitch detector confidence outputs

    Parameters
    ----------
    conf
        2D matrix (number of frames by number of pitch candidates) of deep-learning model outcome
    f
        1D vector of pitch candidate frequencies
    eta_s, optional
        Nonharmonic threshold - pitch estimate with a confidence below this value is considered low confidence, likely a nonharmonic frame, by default 0.25
    eta_v, optional
        Harmonic threshold - pitch estimate with a confidence above this value is considered high confidence, by default 0.75
    rho_o, optional
        Octave cost factor - higher to penalize lower pitch estimate logarithmically more, by default 0.01 (the minimum frequency candidate is penalized by this value)
    rho_oj, optional
        Octave jump cost factor - higher to penalize logarithmically large frame-to-frame frequency jump, by default 0.35 (halving or doubling frequency is penalized by this value)
    vu_high_cost, optional
        Voiced <-> unvoiced transition cost if next estimate is confidently harmonic/nonharmonic, by default 0.5
    vu_low_cost, optional
        voiced <-> unvoiced transition cost if not confidently harmonic/nonharmonic, by default 0.05
    Returns
    -------
        The chosen sequence of frequency bin indices
    """

    fmax = f[-1]  # pitch ceiling
    Su = 1 - conf.max(1)  # unvoiced score
    Sv = conf - rho_o * np.log2(fmax / f)  # voiced score with high frequency penalty

    flog2 = np.log2(f)
    F1, F2 = np.meshgrid(flog2, flog2)
    Rho_vv = rho_oj * np.abs(F1 - F2) / (flog2[-1] - flog2[0])

    # voiced<->unvoiced transition costs
    Tref = conf.max(axis=1, keepdims=True)

    # voiced-to-unvoiced transition requires current frame to have low confidence pitch -> penalize high confidence pitch
    Tvu = np.where(Tref > eta_s, vu_high_cost, vu_low_cost)
    # unvoiced-to-voiced transition requires current frame to have high confidence pitch -> penalize low confidence pitch
    Tuv = np.where(Tref > eta_v, vu_low_cost, vu_high_cost)

    nx, ncands = conf.shape
    Delta = np.zeros((nx, ncands + 1))
    Psi = np.zeros((nx, ncands + 1), int)
    # additional state for nonharmonic

    delta = np.empty((ncands + 1, ncands + 1))
    Dvv = delta[:-1, :-1]
    Dvu = delta[:-1, -1:]
    Duv = delta[-1:, :-1]
    Duu = delta[-1:, -1:]
    for i in range(nx):
        Dlast = Delta[i - 1]  # initial data stored on the last row
        Dvlast = Dlast[:-1].reshape(-1, 1)
        Dulast = Dlast[-1]
        Dvv[:, :] = Dvlast - Rho_vv + Sv[i]  # voiced->voiced
        Dvu[:] = Dvlast - Tvu[i] + Su[i]  # voiced->unvoiced
        Duv[:] = Dulast - Tuv[i] + Sv[i]  # unvoiced->voiced
        Duu[:] = Dulast + Su[i]  # unvoiced->unvoiced
        Psi[i] = idx = np.argmax(delta, axis=0, keepdims=True)
        Delta[i] = np.take_along_axis(delta, idx, axis=0)

    j = np.argmax(Delta[-1])
    return [*reversed([j := psi[j] for psi in Psi[:0:-1]]), j]


def tohertz(
    inputs: NDArray,
    fbins: NDArray,
    center: NDArray | None = None,
    incl_nh: bool = False,
    threshold: float = 0.5,
    nb_average: int = 9,
) -> NDArray:
    # the bin number-to-cents bin_freqs
    bin_freqs = fbins.reshape(1, -1)
    index_delta = np.arange(nb_average).reshape(1, -1)
    offset = nb_average // 2

    start_max = inputs.shape[-1] - index_delta.shape[-1]

    # peak index
    center = np.argmax(inputs, axis=-1) if center is None else np.asarray(center)

    if incl_nh:
        tf = center < len(fbins)
        start = np.clip(center[tf] - offset, 0, start_max).reshape(-1, 1)
        indices = start + index_delta
        weights = np.take_along_axis(inputs[tf, :], indices, axis=1)
    else:
        start = np.clip(center - offset, 0, start_max).reshape(-1, 1)
        indices = start + index_delta
        weights = np.take_along_axis(inputs, indices, axis=1)

    # weighted mean of 9 values
    c = np.take_along_axis(bin_freqs, indices, axis=1)

    product_sum = np.sum(c * weights, axis=1)
    weight_sum = np.sum(weights, axis=1)

    if incl_nh:
        f = np.empty_like(tf, float)
        f[tf] = product_sum / weight_sum
        f[~tf] = 0.0
    else:
        f = product_sum / weight_sum
        if threshold > 0:
            # voice detector
            confidence = np.take_along_axis(inputs, center.reshape(-1, 1), axis=1)
            voiced = confidence > threshold
            f = np.where(voiced, f, 0.0)
            # confidence = np.where(voiced, confidence, 1.0 - confidence)

    return f
