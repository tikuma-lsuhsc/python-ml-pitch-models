"""Abstract Base class to implement short-time analysis process, based on scipy.signal.ShortTimeProcess. """

# Provides typing union operator ``|`` in Python 3.9:
from __future__ import annotations

# Linter does not allow to import ``Generator`` from ``typing`` module:
from abc import ABC, abstractmethod
from functools import cache, lru_cache, cached_property
from typing import get_args, Literal
from numpy.typing import ArrayLike

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from math import ceil

from .utils import PAD_TYPE, cents2freq

__all__ = ["ShortTimeProcess", "ShortTimeStreamProcess"]


#: Allowed values for parameter `padding` of method `ShortTimeProcess.stft()`:
PAD_TYPE = Literal["zeros", "edge", "even", "odd"]


# noinspection PyShadowingNames
class ShortTimeProcess(ABC):
    r"""Abstract base class for short-time time-frequency signal analysis

    .. currentmodule:: scipy.signal.ShortTimeProcess

    The `~ShortTimeProcess.prepare_frames` prepares a set of windows of size `nb_input`, 
    sliding over an input signal by `hop` increments. 

    The p-th window is centered at the time t[p] = p * `delta_t` = p * `hop` * `T` 
    where `T` is  the sampling interval of the input signal. 
    
    This class also generate frequency bins for the analysis in two scaling modes. If
    `f_scale = 'linear'`, the q-th frequency bin is placed at 
    f[q] = q * `delta_f` + `f_min` with `delta_f` is the bin width and `f_min` is the
    frequency of the first bin, both in Hz. If `f_scale = 'log'`, the q-th frequency 
    bin is placed at 
    f[q] = 2**((q * `delta_f` + `f_min`) / 1200) with `delta_f` is the bin width 
    in cents and `f_min` is the frequency of the first bin in cents relative to 1 Hz.

    Due to the convention of time t = 0 being at the first sample of the input
    signal, the STFT values typically have negative time slots. Hence,
    negative indexes like `p_min` or `k_min` do not indicate counting
    backwards from an array's end like in standard Python indexing but being
    left of t = 0.

    More detailed information can be found in the :ref:`tutorial_stft` section
    of the :ref:`user_guide`.

    Abstract Properties
    -------------------
    nb_input
        The input layer size to produce exactly one pitch estimate.
    hop
        The increment in samples, by which the input window is shifted in each step.
    fs
        Sampling frequency of input signal and window. Its relation to the
        sampling interval `T` is ``T = 1 / fs``.
    f_scale
        Classifier frequency bin spacing mode: equispaced in linear or log scale.
    nb_freq_bins
        Number of classifier frequency bins.
    f_min
        Frequency of the lowest classifier bin
    delta_f
        Spacing between classifier frequency bin. In Hz if `f_scale='linear'` or
        in cents if `f_scale='log''.

    Examples
    --------
    The following example shows the magnitude of the STFT of a sine with
    varying frequency :math:`f_i(t)` (marked by a red dashed line in the plot):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import ShortTimeProcess
    >>> from scipy.signal.windows import gaussian
    ...
    >>> T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
    >>> t_x = np.arange(N) * T_x  # time indexes for signal
    >>> f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
    >>> x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal

    The utilized Gaussian window is 50 samples or 2.5 s long. The parameter
    ``nb_freq_bins=200`` in `ShortTimeProcess` causes the spectrum to be oversampled
    by a factor of 4:

    >>> g_std = 8  # standard deviation for Gaussian window in samples
    >>> w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    >>> SFT = ShortTimeProcess(w, hop=10, fs=1/T_x, nb_freq_bins=200, scale_to='magnitude')
    >>> Sx = SFT.stft(x)  # perform the STFT

    In the plot, the time extent of the signal `x` is marked by vertical dashed
    lines. Note that the SFT produces values outside the time range of `x`. The
    shaded areas on the left and the right indicate border effects caused
    by  the window slices in that area not fully being inside time range of
    `x`:

    >>> fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    >>> t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    >>> ax1.set_title(rf"STFT ({SFT.nb_input*SFT.T:g}$\,s$ Gaussian window, " +
    ...               rf"$\sigma_t={g_std*SFT.T}\,$s)")
    >>> ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
    ...                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
    ...         ylabel=f"Freq. $f$ in Hz ({SFT.nb_freq_bins} bins, " +
    ...                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
    ...         xlim=(t_lo, t_hi))
    ...
    >>> im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
    ...                  extent=SFT.extent(N), cmap='viridis')
    >>> ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    >>> fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    ...
    >>> # Shade areas where window slices stick out to the side:
    >>> for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
    ...                  (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    ...     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    >>> for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    ...     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    >>> ax1.legend()
    >>> fig1.tight_layout()
    >>> plt.show()

    Reconstructing the signal with the `~ShortTimeProcess.istft` is
    straightforward, but note that the length of `x1` should be specified,
    since the SFT length increases in `hop` steps:

    >>> SFT.invertible  # check if invertible
    True
    >>> x1 = SFT.istft(Sx, k1=N)
    >>> np.allclose(x, x1)
    True

    It is possible to calculate the SFT of signal parts:

    >>> p_q = SFT.nearest_k_p(N // 2)
    >>> Sx0 = SFT.stft(x[:p_q])
    >>> Sx1 = SFT.stft(x[p_q:])

    When assembling sequential STFT parts together, the overlap needs to be
    considered:

    >>> p0_ub = SFT.upper_border_begin(p_q)[1] - SFT.p_min
    >>> p1_le = SFT.lower_border_end[1] - SFT.p_min
    >>> Sx01 = np.hstack((Sx0[:, :p0_ub],
    ...                   Sx0[:, p0_ub:] + Sx1[:, :p1_le],
    ...                   Sx1[:, p1_le:]))
    >>> np.allclose(Sx01, Sx)  # Compare with SFT of complete signal
    True

    It is also possible to calculate the `itsft` for signal parts:

    >>> y_p = SFT.istft(Sx, N//3, N//2)
    >>> np.allclose(y_p, x[N//3:N//2])
    True

    """

    _hop: int

    def __init__(self, hop: int):
        if not (isinstance(hop, int) and hop >= 0):
            raise ValueError(f"{hop=} must be a positive integer")
        super().__init__()
        self._hop = hop

    @property
    def hop(self) -> int:
        """Time increment in signal samples for sliding window.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        nb_input: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        nb_freq_bins: Length of input for the FFT used - may be larger than `nb_input`.
        T: Sampling interval of input signal and of the window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeProcess: Class this property belongs to.
        """
        return self._hop

    @property
    def T(self) -> float:
        """Sampling interval of input signal and of the window.

        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeProcess: Class this property belongs to.
        """
        return 1 / self.fs

    @property
    @abstractmethod
    def fs(self) -> float:
        """Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        ShortTimeProcess: Class this property belongs to.
        """

    @property
    @abstractmethod
    def f_scale(self) -> Literal["linear", "log"]:
        """Classifier frequency bin spacing mode: equispaced in linear or log scale."""

    @property
    @abstractmethod
    def nb_freq_bins(self) -> int:
        """Number of classifier frequency bins."""

    @property
    @abstractmethod
    def f_min(self) -> float:
        """Frequency of the lowest classifier bin."""

    @property
    @abstractmethod
    def delta_f(self) -> float:
        """Spacing between classifier frequency bin. In Hz if `f_scale='linear'` or
        in cents if `f_scale='log''."""

    @property
    @abstractmethod
    def nb_input(self) -> int:
        """Size of the model input"""

    @property
    def m_num_mid(self) -> int:
        """Center index of window `win`.

        For odd `nb_input`, ``(nb_input - 1) / 2`` is returned and
        for even `nb_input` (per definition) ``nb_input / 2`` is returned.

        See Also
        --------
        nb_input: Number of samples in window `win`.
        nb_freq_bins: Length of input for the FFT used - may be larger than `nb_input`.
        hop: ime increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeProcess: Class this property belongs to.
        """
        return self.nb_input // 2

    @cache
    def _pre_padding(self) -> tuple[int, int]:
        """Smallest signal index and slice index due to padding.

        Since, per convention, for time t=0, n,q is zero, the returned values
        are negative or zero.
        """

        # move window to the left until the overlap with t >= 0 vanishes:
        return -self.m_num_mid, 0

    @property
    def k_min(self) -> int:
        """The smallest possible signal index of the STFT.

        `k_min` is the index of the left-most non-zero value of the lowest
        slice `p_min`. Since the zeroth slice is centered over the zeroth
        sample of the input signal, `k_min` is never positive.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeProcess: Class this property belongs to.
        """
        return self._pre_padding()[0]

    @property
    def p_min(self) -> int:
        """The smallest possible slice index.

        `p_min` is the index of the left-most slice, where the window still
        sticks into the signal, i.e., has non-zero part for t >= 0.
        `k_min` is the smallest index where the window function of the slice
        `p_min` is non-zero.

        Since, per convention the zeroth slice is centered at t=0,
        `p_min` <= 0 always holds.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeProcess: Class this property belongs to.
        """
        return self._pre_padding()[1]

    @lru_cache(maxsize=256)
    def _post_padding(self, n: int) -> tuple[int, int]:
        """Largest signal index and slice index due to padding."""

        # move window to the right until the overlap for t < t[n] vanishes:
        q1 = n // self.hop  # last slice index with t[p1] <= t[n]
        k1 = q1 * self.hop - self.m_num_mid
        return k1 + self.nb_input, q1 + 1

    def k_max(self, n: int) -> int:
        """First sample index after signal end not touched by a time slice.

        `k_max` - 1 is the largest sample index of the slice `p_max` for a
        given input signal of `n` samples.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeProcess: Class this method belongs to.
        """
        return self._post_padding(n)[0]

    def p_max(self, n: int) -> int:
        """Index of first non-overlapping upper time slice for `n` sample
        input.

        Note that center point t[p_max] = (p_max(n)-1) * `delta_t` is typically
        larger than last time index t[n-1] == (`n`-1) * `T`. The upper border
        of samples indexes covered by the window slices is given by `k_max`.
        Furthermore, `p_max` does not denote the number of slices `p_num` since
        `p_min` is typically less than zero.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_min: The smallest possible slice index.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeProcess: Class this method belongs to.
        """
        return self._post_padding(n)[1]

    def p_num(self, n: int) -> int:
        """Number of time slices for an input signal with `n` samples.

        It is given by `p_num` = `p_max` - `p_min` with `p_min` typically
        being negative.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeProcess: Class this method belongs to.
        """
        return self.p_max(n) - self.p_min

    @cached_property
    def lower_border_end(self) -> tuple[int, int]:
        """First signal index and first slice index unaffected by pre-padding.

        Describes the point where the window does not stick out to the left
        of the signal domain.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeProcess: Class this property belongs to.
        """
        q_ = ceil(self.m_num_mid / self.hop)  # first window with original samples
        k_ = q_ * self.hop
        return k_, q_

    @lru_cache(maxsize=256)
    def upper_border_begin(self, n: int) -> tuple[int, int]:
        """First signal index and first slice index affected by post-padding.

        Describes the point where the window does begin stick out to the right
        of the signal domain.
        A detailed example is given :ref:`tutorial_stft_sliding_win` section
        of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeProcess: Class this method belongs to.
        """
        q1 = (n - self.nb_input) // self.hop + 1
        return q1 * self.hop - self.m_num_mid, q1

    @property
    def delta_t(self) -> float:
        """Time increment of pitch estimates .

        The time increment `delta_t` = `T` * `hop` represents the sample
        increment `hop` converted to time based on the sampling interval `T`.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        hop: Hop size in signal samples for sliding window.
        t: Times of STFT for an input signal with `n` samples.
        T: Sampling interval of input signal and window `win`.
        ShortTimeProcess: Class this property belongs to
        """
        return self.T * self.hop

    def p_range(
        self, n: int, p0: int | None = None, p1: int | None = None
    ) -> tuple[int, int]:
        """Determine and validate slice index range.

        Parameters
        ----------
        n : int
            Number of samples of input signal, assuming t[0] = 0.
        p0 : int | None
            First slice index. If 0 then the first slice is centered at t = 0.
            If ``None`` then `p_min` is used. Note that p0 may be < 0 if
            slices are left of t = 0.
        p1 : int | None
            End of interval (last value is p1-1).
            If ``None`` then `p_max(n)` is used.


        Returns
        -------
        p0_ : int
            The fist slice index
        p1_ : int
            End of interval (last value is p1-1).

        Notes
        -----
        A ``ValueError`` is raised if ``p_min <= p0 < p1 <= p_max(n)`` does not
        hold.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        upper_border_begin: Where post-padding effects start.
        ShortTimeProcess: Class this property belongs to.
        """
        p_max = self.p_max(n)  # shorthand
        p0_ = self.p_min if p0 is None else p0
        p1_ = p_max if p1 is None else p1 if p1 >= 0 else p1 + p_max
        if not (self.p_min <= p0_ < p1_ <= p_max):
            raise ValueError(
                f"Invalid Parameter {p0=}, {p1=}, i.e., "
                + f"{self.p_min=} <= p0 < p1 <= {p_max=} "
                + f"does not hold for signal length {n=}!"
            )
        return p0_, p1_

    @lru_cache(maxsize=1)
    def t(
        self, n: int, p0: int | None = None, p1: int | None = None, k_offset: int = 0
    ) -> np.ndarray:
        """Times of STFT for an input signal with `n` samples.

        Returns a 1d array with times of the `~ShortTimeProcess.stft` values with
        the same  parametrization. Note that the slices are
        ``delta_t = hop * T`` time units apart.

         Parameters
        ----------
        n
            Number of sample of the input signal.
        p0
            The first element of the range of slices to calculate. If ``None``
            then it is set to :attr:`p_min`, which is the smallest possible
            slice.
        p1
            The end of the array. If ``None`` then `p_max(n)` is used.
        k_offset
            Index of first sample (t = 0) in `x`.


        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        nearest_k_p: Nearest sample index k_p for which t[k_p] == t[p] holds.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        ShortTimeProcess: Class this method belongs to.
        """
        p0, p1 = self.p_range(n, p0, p1)
        return np.arange(p0, p1) * self.delta_t + k_offset * self.T

    def nearest_k_p(self, k: int, left: bool = True) -> int:
        """Return nearest sample index k_p for which t[k_p] == t[p] holds.

        The nearest next smaller time sample p (where t[p] is the center
        position of the window of the p-th slice) is p_k = k // `hop`.
        If `hop` is a divisor of `k` than `k` is returned.
        If `left` is set than p_k * `hop` is returned else (p_k+1) * `hop`.

        This method can be used to slice an input signal into chunks for
        calculating the STFT and iSTFT incrementally.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeProcess: Class this method belongs to.
        """
        p_q, remainder = divmod(k, self.hop)
        if remainder == 0:
            return k
        return p_q * self.hop if left else (p_q + 1) * self.hop

    @cached_property
    def f(self) -> np.ndarray:
        """Frequencies values of the model's pitch classifer bins.

        A 1d array of length `nb_freq_bins` with `delta_f` spaced entries is returned.

        See Also
        --------
        delta_f: Width of the frequency bins of the classifer output.
        nb_freq_bins: Number of points along the frequency axis.
        ShortTimeProcess: Class this property belongs to.
        """

        is_linear = self.f_scale == "linear"
        f = np.arange(self.nb_freq_bins) * self.delta_f + self.f_min
        return f if is_linear else cents2freq(f)

    @lru_cache(maxsize=256)
    def extent(
        self,
        n: int,
        p0: int | None = None,
        p1: int | None = None,
        k_offset: int = 0,
        axes_seq: Literal["tf", "ft"] = "tf",
        center_bins: bool = False,
    ) -> tuple[float, float, float, float]:
        """Return minimum and maximum values time-frequency values.

        A tuple with four floats  ``(t0, t1, f0, f1)`` for 'tf' and
        ``(f0, f1, t0, t1)`` for 'ft' is returned describing the corners
        of the time-frequency domain of the `~ShortTimeProcess.stft`.
        That tuple can be passed to `matplotlib.pyplot.imshow` as a parameter
        with the same name.

        Parameters
        ----------
        n : int
            Number of samples in input signal.
        axes_seq : {'tf', 'ft'}
            Return time extent first and then frequency extent or vice-versa.
        center_bins: bool
            If set (default ``False``), the values of the time slots and
            frequency bins are moved from the side the middle. This is useful,
            when plotting the `~ShortTimeProcess.stft` values as step functions,
            i.e., with no interpolation.

        See Also
        --------
        :func:`matplotlib.pyplot.imshow`: Display data as an image.
        :class:`scipy.signal.ShortTimeProcess`: Class this method belongs to.
        """
        if axes_seq not in ("tf", "ft"):
            raise ValueError(f"Parameter {axes_seq=} not in ['tf', 'ft']!")

        is_log = self.f_scale == "log"

        q0, q1 = 0, self.nb_freq_bins
        p0, p1 = self.p_range(n, p0, p1)

        # np.arange(p0, p1) * self.delta_t + k_offset * self.T
        if center_bins:
            t0, t1 = self.delta_t * p0, self.delta_t * p1
            f0, f1 = self.delta_f * q0, self.delta_f * q1
        else:
            t0, t1 = self.delta_t * (p0 - 0.5), self.delta_t * (p1 - 0.5)
            f0, f1 = self.delta_f * (q0 - 0.5), self.delta_f * (q1 - 0.5)

        if k_offset:
            t_off = k_offset * self.T
            t0, t1 = t0 + t_off, t1 + t_off

        f_min = self.f_min
        f0, f1 = f0 + f_min, f1 + f_min

        if is_log:
            f0, f1 = cents2freq(f0), cents2freq(f1)

        return (t0, t1, f0, f1) if axes_seq == "tf" else (f0, f1, t0, t1)

    def pad_signal(
        self, x: np.ndarray, k_off: int, p0: int, p1: int, padding: PAD_TYPE
    ) -> np.ndarray:
        """Generate padded signal slices along last axis of `x`.

        The analysis windows of length nb_input is sled over the last axis of the input
        signal x by hop increments.

        Let k0 = nb_input//2 and k1 = nb_input - k0. Then, the p-th window is spanning
        the signal indices: [p * hop - k0 + k_off, p*hop + k1 + k_off).

            Args:
                x
                    signal array (time axis = last axis)
                nb_input
                    frame size
                hop
                    increment in signal samples for sliding window
                p0
                    First window index. Defaults to 0.
                p1
                    End of interval (last window is p1-1).
                    Defaults to None to be the latest window with its center within
                    the signal duration.
                k_off
                    Index of first sample (t = 0) in `x`. Defaults to 0.
                padding
                    signal padding mode. Defaults to "zeros".

            Returns:
                padded/trimmed signal to cover windows p0 to p1.

        """

        if padding not in (padding_types := get_args(PAD_TYPE)):
            raise ValueError(f"Parameter {padding=} not in {padding_types}!")
        pad_kws: dict[str, dict] = {  # possible keywords to pass to np.pad:
            "zeros": dict(mode="constant", constant_values=(0, 0)),
            "edge": dict(mode="edge"),
            "even": dict(mode="reflect", reflect_type="even"),
            "odd": dict(mode="reflect", reflect_type="odd"),
        }  # typing of pad_kws is needed to make mypy happy

        n, n1 = x.shape[-1], (p1 - p0) * self.hop
        k0 = p0 * self.hop - self.m_num_mid + k_off  # start sample
        k1 = k0 + n1 + self.nb_input - 1  # end sample

        i0, i1 = max(k0, 0), min(k1, n)  # indexes to shorten x
        # dimensions for padding x:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(-min(k0, 0), max(k1 - n, 0))]

        return np.pad(x[..., i0:i1], pad_width, **pad_kws[padding])

    def prepare_frames(
        self,
        x: ArrayLike,
        p0: int | None = None,
        p1: int | None = None,
        k_off: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> np.ndarray:
        """Create a normalized sliding window view into the padded signal.

        Args:
            x
                signal array (time axis = last axis)
            nb_input
                frame size
            hop
                increment in signal samples for sliding window
            p0
                First window index. Defaults to 0.
            p1
                End of interval (last window is p1-1).
                Defaults to None to be the latest window with its center within
                the signal duration.
            k_off
                Index of first sample (t = 0) in `x`. Defaults to 0.
            padding
                signal padding mode. Defaults to "zeros".
            axis
                The axis of `x` over which to run the model along.
                If not given, the last axis is used.

        Returns:
            np.ndarray: array view containing windows from p0 to p1. The frame axis
            is the second from the last dimension.
        """

        n = x.shape[axis]
        if not (n >= (m2p := self.nb_input - self.m_num_mid)):
            e_str = f"{len(x)=}" if x.ndim == 1 else f"of {axis=} of {x.shape}"
            raise ValueError(f"{e_str} must be >= ceil(nb_input/2) = {m2p}!")

        if x.ndim > 1:  # motivated by the NumPy broadcasting mechanisms:
            x = np.moveaxis(x, axis, -1)

        # determine slice index range:
        p0, p1 = self.p_range(n, p0, p1)

        frames = sliding_window_view(
            self.pad_signal(x, k_off, p0, p1, padding), self.nb_input, -1
        )[..., :: self.hop, :]

        # normalize each frame -- this is expected by the model
        std = np.std(frames, axis=-1, keepdims=True)
        std[std == 0.0] = 1
        return (frames - np.mean(frames, axis=-1, keepdims=True)) / std


class ShortTimeStreamProcess(ShortTimeProcess):

    @property
    @abstractmethod
    def native_hop(self) -> int:
        """Time increment in signal samples imposed by the stream process."""

    _hop: int | None  # user-specified hop or None if operate at the native hop

    @property
    def hop(self) -> int:
        """Time increment in signal samples for sliding window.

        This attribute is read only.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        nb_input: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        nb_freq_bins: Length of input for the FFT used - may be larger than `nb_input`.
        T: Sampling interval of input signal and of the window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeProcess: Class this property belongs to.
        """

        return self._hop or self.native_hop

    def prepare_frames(
        self,
        x: ArrayLike,
        p0: int | None = None,
        p1: int | None = None,
        k_off: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> np.ndarray:
        """Create a normalized sliding window view into the padded signal.

        Args:
            x
                signal array (time axis = last axis)
            nb_input
                frame size
            hop
                increment in signal samples for sliding window
            p0
                First window index. Defaults to 0.
            p1
                End of interval (last window is p1-1).
                Defaults to None to be the latest window with its center within
                the signal duration.
            k_off
                Index of first sample (t = 0) in `x`. Defaults to 0.
            padding
                signal padding mode. Defaults to "zeros".
            axis
                The axis of `x` over which to run the model along.
                If not given, the last axis is used.

        Returns:
            np.ndarray: array view containing windows from p0 to p1. The frame axis
            is the second from the last dimension.
        """

        if self._hop:
            # batch processing
            return super().prepare_frames(x, p0, p1, k_off, padding, axis)

        # fully-convolutional single-batch processing

        n = x.shape[axis]
        if not (n >= (m2p := self.nb_input - self.m_num_mid)):
            e_str = f"{len(x)=}" if x.ndim == 1 else f"of {axis=} of {x.shape}"
            raise ValueError(f"{e_str} must be >= ceil(nb_input/2) = {m2p}!")

        if x.ndim > 1:  # motivated by the NumPy broadcasting mechanisms:
            x = np.moveaxis(x, axis, -1)

        # normalize moving-frame
        nb_input = self.nb_input
        m_num_mid = self.m_num_mid
        pad_width = [(0, 0)] * (x.ndim - 1) + [(m_num_mid, nb_input - m_num_mid - 1)]
        xpadded = np.pad(x, pad_width, mode="reflect", reflect_type="even")
        frames = sliding_window_view(xpadded, nb_input, -1)
        mean = np.mean(frames, axis=-1)
        std = np.std(frames, axis=-1)
        std[std == 0.0] = 1
        x = (x - mean) / std

        # determine slice index range:
        p0, p1 = self.p_range(n, p0, p1)

        return self.pad_signal(x, k_off, p0, p1, padding)
