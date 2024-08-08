
``ml-pitch-models`` - A collection of ML-based pitch detector Python Tensorflow models
**************************************************************************************


Description
===========

This Python package features two deep-learning pitch detection models.

1. ``CrepeModel`` - CREPE: deep convolutional neural network (CNN)
   model (5 pretrained weights)

   J. W. Kim, J. Salamon, P. Li, J. P. Bello. “CREPE: A Convolutional
   Representation for Pitch Estimation”, Proc. IEEE ICASSP, 2018. doi:
   10.1109/ICASSP.2018.8461329

   GitHub Repository: https://github.com/marl/crepe

2. ``FcnF0Model`` - FCN-F0: fully convolutional network (FCN) model (3
   pretrained weights)

   L. Ardaillon and A. Roebel, “Fully-Convolutional Network for Pitch
   Estimation of Speech Signals”, Proc. Interspeech, 2019. doi:
   10.21437/Interspeech.2019-2815

   GitHub Repository: https://github.com/ardaillon/FCN-f0

While both of these models are open-sourced by the respective authors,
including their pre-trained weights, this package distributes only the
models with a convenient mechanism to also distribute pretrained
weights, minimizing the footprint of the package.


Installation
============

To install just the models *without* any pretrained weights:

.. code:: bash

   pip install ml-pitch-models

Each set of pretrained weights is available as a separate PyPI
package, and they can be installed using *pip*’s the optional extra
dependency specifiers.

Based on Ardaillon and Roebel’s choice, the (current) default model
for *ml_pitch_models.predict()* is the *FCN-993* model, and to include
the pretrained weights of the default model:

.. code:: bash

   pip install ml-pitch-models[DEFAULT_WEIGHTS]

To install individual pretrained weights, use the following extra
dependency keywords or a combination thereof.

.. code:: bash

   pip install ml-pitch-models[CREPE_FULL_WEIGHTS]
   pip install ml-pitch-models[CREPE_LARGE_WEIGHTS]
   pip install ml-pitch-models[CREPE_MEDIUM_WEIGHTS]
   pip install ml-pitch-models[CREPE_SMALL_WEIGHTS]
   pip install ml-pitch-models[CREPE_TINY_WEIGHTS]
   pip install ml-pitch-models[FCN_1953_WEIGHTS]
   pip install ml-pitch-models[FCN_993_WEIGHTS]
   pip install ml-pitch-models[FCN_929_WEIGHTS]


Tensorflow Installation
-----------------------

This package only only installs the base *tensorflow* package as its
dependency. To enable GPU acceleration (in Linux), also run:

.. code:: bash

   pip install tensorflow[and-cuda]

Read https://www.tensorflow.org/install/pip for more details. Users
are responsible to enable the GPU support in tensorflow before running
the pitch prediction.


Basic Usage
===========

Assume ``x`` is an 1D Numpy array, containing a signal sampled at 8
kHz (``fs=8000``)

.. code:: python

   import ml_pitch_models

   t, f0, conf = ml_pitch_models.predict(fs, x)

computes the pitch estimates ``f0`` with the default model
``fcn_993``. The estimates are generated with a sliding window of size
993 An ``f0`` value of zero indicates that no pitch was detected in
that window.

The timestamps ``t`` is a sequence of the timestamps of the middle of
the window. By default, ``t[0]=0`` and ``t[-1]`` is the last window
with more than half of its bins filled.

The last output ``conf`` is the confidence levels of the estimates.
This is the maximum value of the DL classifier layer if pitch is
detected or ``1-max`` if no pitch is detected.

The third argument of ``ml_pitch_models.predict`` is the DL model to
use. To find the available pretrained models, run

.. code:: python

   ml_pitch_models.available_models() # returns a list of model names

Possible model names are: ``"crepe_full"``, ``"crepe_large"``,
``"crepe_medium"``, ``"crepe_small"``, ``"crepe_tiny"``,
``"fcn_1953"``, ``"fcn_929"``, and ``"fcn_993"``. Only the installed
pretrained models would be listed.

Note that the pretrained CREPE model and the 8-kHz signal ``x`` is not
compatible because the CREPE models were pretrained with 16-kHz
signals. In other words,

.. code:: python

   t, f0, conf = ml_pitch_models.predict(fs, x, 'crepe_large')

will throw a ``ValueError`` exception because the signal sampling rate
``fs`` does not match the model’s input sampling rate (16 kHz). The
signal must first be interpolated by 2 to 16 kHz to run ``x`` through
a CREPE pretrained model. For example, you can use
`scipy.signal.resample_poly
<https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly>`_.


``hop`` option
--------------

The pitches are estimated over a sliding window, producing estimates
at a ``hop`` interval, specified in samples. The key difference
between the CREPE and FCN-F0 models is that the latter can run the
pitch detection more efficiently with a window hop size that is
imposed by the model (``native_hop``) in fully convolutional mode of
operation. The FCN-F0 models default to the fully convolutional mode
if ``hop`` argument is omitted. To explicitly specify the fully
convolutional operation, set ``hop`` argument to ``'native'`` or
``0``. Otherwise, setting ``hop`` to a positive integer sets the
models to operate in a batch mode with the estimate interval
``hop/fs`` seconds where ``fs`` is the sampling rate. The CREPE models
always operate in the batch mode and omission of the ``hop`` argument
defaults to a 10-ms or 160-sample interval.

Examples:

.. code:: python

   ml_pitch_models.predict(fs, x, 'fcn_993', hop='native') # fully convolutional mode, 1-ms (8-sample) interval
   ml_pitch_models.predict(fs, x, 'fcn_929', hop='native') # fully convolutional mode, 0.5-ms (4-sample) interval
   ml_pitch_models.predict(fs, x, 'fcn_929', hop=400) # 50-ms hop size
   ml_pitch_models.predict(fs, x, 'crepe_tiny', hop=400) # 25-ms hop size

Note that the same ``hop`` value results in a different interval due
to the difference in the sampling rates between the CREPE and FCN-F0
models.


``postprocessor`` and ``postprocessor_kws`` options
---------------------------------------------------

These options enables a dynamical-programming postprocessers. There
are currently two options:

*  ``'viterbi'`` - Adoptation of Praat’s postprocessor to find the
   frequency transitions based on the ML model’s confidence vectors.
   It also detects nonharmonic frames (frames with low confidence
   level). More information will follow.

*  ``'hmm'`` - The original postprocessor in CREPE and FCN-F0
   repositories referenced above. It enforces the successive
   frequencies to be close, preventing a large jump.

Examples:

.. code:: python

   ml_pitch_models.predict(fs, x, 'fcn_993', postprocessor='viterbi')
   ml_pitch_models.predict(fs, x, 'fcn_993', postprocessor='hmm')


Simultaneous processing of multiple signals
-------------------------------------------

Both models supports processing multiple signals (of the same length)
at once. The signal arrays must be stack in “rows”. Suppose that we
have 3 1D signals: ``x0```, ``x1``, and ``x2``:

.. code:: python

   import numpy as np

   t, f0, conf = ml_pitch_models.predict(fs, np.stack([x0, x1, x2], axis=0))

Then, ``f0`` and ``conf`` are 2D arrays with 3 rows, corresponding to
the input signals.

Assuming that the system is not memory constrained, this yields in a
faster execution as the model is only constructed once.


API Reference
=============

**ml_pitch_models.predict(x: ArrayLike, fs: int, model:
Literal['crepe_full', 'crepe_large', 'crepe_medium', 'crepe_small',
'crepe_tiny', 'fcn_1953', 'fcn_929', 'fcn_993'] | FcnF0Model |
CrepeModel = 'fcn_993', hop: int | None = None, harmonic_threshold:
float | None = None, postprocessor: Literal['viterbi', 'hmm'] | None =
None, postprocessor_kws: dict | None = None, axis: int = -1, **kwargs)
-> tuple[ndarray, ndarray, ndarray]**

   Generates pitch predictions for the input signal.

   :Parameters:
      *  **x** (ArrayLike) – Input signal(s). For a higher dimensional
         array, the pitch is detected along the last dimension.

      *  **fs** (``int``) – Input signal sampling rate in
         Samples/second. This must match model.fs.

      *  **model** (``Union``[``Literal``[``'crepe_full'``,
         ``'crepe_large'``, ``'crepe_medium'``, ``'crepe_small'``,
         ``'crepe_tiny'``, ``'fcn_1953'``, ``'fcn_929'``,
         ``'fcn_993'``], `FcnF0Model <#ml_pitch_models.FcnF0Model>`_,
         `CrepeModel <#ml_pitch_models.CrepeModel>`_], default:
         ``'fcn_993'``) – Pitch detection deep-learning model.

      *  **hop** (``int`` | ``None``, default: ``None``) – The
         increment in signal samples, by which the window is shifted
         in each step for frame-wise processing. If None, hop is set
         to (roughly) 10 ms for CREPE models and ‘native’ for FCN-F0
         models

      *  **harmonic_threshold** (``float`` | ``None``, default:
         ``None``) – Harmonic detection threshold on the classifier
         confidence level. If confidence level is below this
         threshold, f0=0 is returned, indicating the frame has no
         harmonic content.

      *  **postprocessor** (``Optional``[``Literal``[``'viterbi'``,
         ``'hmm'``]], default: ``None``) – Specify to enable the
         dynamic programming postprocessor to select the frequency
         transitions

      *  **postprocessor_kws** (``dict`` | ``None``, default:
         ``None``) – Specify the options for the postprocessor.

      *  **hop** – The increment in signal samples, by which the
         window is shifted in each step for frame-wise processing. If
         None, hop size of (roughly) 10 ms is used. For stream
         processing, this argument is ignored and the native hop size
         (self.native_hop) of the model is used instead.

      *  **p0** – The first element of the range of slices to
         calculate. If None then it is set to p_min, which is the
         smallest possible slice.

      *  **p1** – The end of the array. If None then the largest
         possible slice is used.

      *  **k_offset** – Index of first sample (t = 0) in x.

      *  **padding** – Kind of values which are added, when the
         sliding window sticks out on either the lower or upper end of
         the input x. Zeros are added if the default ‘zeros’ is set.
         For ‘edge’ either the first or the last value of x is used.
         ‘even’ pads by reflecting the signal on the first or last
         sample and ‘odd’ additionally multiplies it with -1.

      *  **batch_size** – Number of samples per batch. If unspecified,
         *batch_size* will default to 32. Do not specify the
         *batch_size* if your data is in the form of dataset,
         generators, or *keras.utils.PyDataset* instances (since they
         generate batches).

      *  **verbose** – *“auto”*, 0, 1, or 2. Verbosity mode. 0 =
         silent, 1 = progress bar, 2 = single line. *“auto”* becomes 1
         for most cases. Note that the progress bar is not
         particularly useful when logged to a file, so *verbose=2* is
         recommended when not running interactively (e.g. in a
         production environment). Defaults to *“auto”*.

      *  **steps** – Total number of steps (batches of samples) before
         declaring the prediction round finished. Ignored with the
         default value of *None*. If *x* is a *tf.data.Dataset* and
         *steps* is *None*, *predict()* will run until the input
         dataset is exhausted.

      *  **callbacks** – List of *keras.callbacks.Callback* instances.
         List of callbacks to apply during prediction.

   :Returns:
      If postprocessor is assigned:
         *  t: timestamps

         *  f0: predicted pitches

      If self.return_f0 is true:
         *  t: timestamps

         *  f0: predicted pitches

         *  confidence: pitch prediction confidences

      If self.return_f0 is false:
         *  t: timestamps

         *  2D array of a sequence of confidence levels of all
            frequency bins

   :Return type:
      ``tuple``[``ndarray``, ``ndarray``, ``ndarray``]

**ml_pitch_models.load_model(model: Literal['crepe_full',
'crepe_large', 'crepe_medium', 'crepe_small', 'crepe_tiny',
'fcn_1953', 'fcn_929', 'fcn_993'] = 'fcn_993', **kwargs) ->
`FcnF0Model <#ml_pitch_models.FcnF0Model>`_ | `CrepeModel
<#ml_pitch_models.CrepeModel>`_**

   Load pretrained pitch estimation model.

   :Parameters:
      *  **model** (``Literal``[``'crepe_full'``, ``'crepe_large'``,
         ``'crepe_medium'``, ``'crepe_small'``, ``'crepe_tiny'``,
         ``'fcn_1953'``, ``'fcn_929'``, ``'fcn_993'``], default:
         ``'fcn_993'``) – Pitch detection deep-learning model.

      *  ****kwargs** – Passed to the model constructor

   :Returns:
      Model object

   :Return type:
      `FcnF0Model <#ml_pitch_models.FcnF0Model>`_ | `CrepeModel
      <#ml_pitch_models.CrepeModel>`_

**class ml_pitch_models.CrepeModel(*layers: tuple[LayerInfo],
weights_file: str | None = None, hop: int | None = None, return_f0:
bool = False, harmonic_threshold: float | None = None, postprocessor:
Literal['viterbi', 'hmm'] | None = None, postprocessor_kws: dict |
None = None, dropout: float = 0.25)**

   CREPE pitch estimation model

   :Parameters:
      *  ***layers** (``tuple``[``LayerInfo``]) – Variable length
         argument list to define CNN layers.

      *  **weights_file** (``str`` | ``None``, default: ``None``) –
         path to the weights file to load. It can either be a
         .weights.h5 file or a legacy .h5 weights file. Defaults to
         None.

      *  **hop** (``int`` | ``None``, default: ``None``) – The
         increment in signal samples, by which the window is shifted
         in each step for frame-wise processing. If None, hop size of
         (roughly) 10 ms is used

      *  **return_f0** (``bool``, default: ``False``) – True to return
         pitch estimates in Hz. Defaults to False to return classifier
         output.

      *  **framewise** – True to transform the input to a sequence of
         sliding window frames. This option must be True or None for
         CrepeModel. Defaults to True.

      *  **harmonic_threshold** (``float`` | ``None``, default:
         ``None``) – Classifier output threshold to detect voice.
         Defaults to None (uses the class default of 0.5.).

      *  **dropout** (``float``, default: ``0.25``) – Dropout rate
         (training only). Defaults to 0.25.

   **predict(x: ArrayLike, fs: int, p0: int = 0, p1: int | None =
   None, k_offset: int = 0, padding: Literal['zeros', 'edge', 'even',
   'odd'] = 'zeros', axis: int = -1, **kwargs) -> tuple[ndarray,
   ndarray] | ndarray**

      Generates pitch predictions for the input signal.

      Computation is done in batches. This method is designed for
      batch processing of large numbers of inputs. It is not intended
      for use inside of loops that iterate over your data and process
      small numbers of inputs at a time.

      For small numbers of inputs that fit in one batch, directly use
      *__call__()* for faster execution, e.g., *model(x)*, or
      *model(x, training=False)* if you have layers such as
      *BatchNormalization* that behave differently during inference.

      :Parameters:
         *  **x** (ArrayLike) – Input signal(s). For a higher
            dimensional array, the pitch is detected along the last
            dimension.

         *  **fs** (``int``) – Input signal sampling rate in
            Samples/second. This must match model.fs.

         *  **p0** (``int``, default: ``0``) – The first element of
            the range of slices to calculate. If None then it is set
            to p_min, which is the smallest possible slice.

         *  **p1** (``int`` | ``None``, default: ``None``) – The end
            of the array. If None then the largest possible slice is
            used.

         *  **k_offset** (``int``, default: ``0``) – Index of first
            sample (t = 0) in x.

         *  **padding** (``Literal``[``'zeros'``, ``'edge'``,
            ``'even'``, ``'odd'``], default: ``'zeros'``) – Kind of
            values which are added, when the sliding window sticks out
            on either the lower or upper end of the input x. Zeros are
            added if the default ‘zeros’ is set. For ‘edge’ either the
            first or the last value of x is used. ‘even’ pads by
            reflecting the signal on the first or last sample and
            ‘odd’ additionally multiplies it with -1.

         *  **axis** (``int``, default: ``-1``) – The axis of *x* over
            which to run the model along. If not given, the last axis
            is used.

         *  **batch_size** – Number of samples per batch. If
            unspecified, *batch_size* will default to 32. Do not
            specify the *batch_size* if your data is in the form of
            dataset, generators, or *keras.utils.PyDataset* instances
            (since they generate batches).

         *  **verbose** – *“auto”*, 0, 1, or 2. Verbosity mode. 0 =
            silent, 1 = progress bar, 2 = single line. *“auto”*
            becomes 1 for most cases. Note that the progress bar is
            not particularly useful when logged to a file, so
            *verbose=2* is recommended when not running interactively
            (e.g. in a production environment). Defaults to *“auto”*.

         *  **steps** – Total number of steps (batches of samples)
            before declaring the prediction round finished. Ignored
            with the default value of *None*. If *x* is a
            *tf.data.Dataset* and *steps* is *None*, *predict()* will
            run until the input dataset is exhausted.

         *  **callbacks** – List of *keras.callbacks.Callback*
            instances. List of callbacks to apply during prediction.

      :Returns:
         If postprocessor is assigned:
            *  f0: predicted pitches

         If self.return_f0 is true:
            *  f0: predicted pitches

            *  confidence: pitch prediction confidences

         If self.return_f0 is false:
            *  2D array of a sequence of confidence levels of all
               frequency bins

      :Return type:
         ``tuple``[``ndarray``, ``ndarray``] | ``ndarray``

**class ml_pitch_models.FcnF0Model(*layers: tuple[LayerInfo],
weights_file: str | None = None, hop: int | None | Literal['native'] =
'native', return_f0: bool = False, harmonic_threshold: float | None =
None, postprocessor: Literal['viterbi', 'hmm'] | None = None,
postprocessor_kws: dict | None = None, dropout: float = 0.25)**

   **predict(x: ArrayLike, fs: int, p0: int | None = None, p1: int |
   None = None, k_offset: int = 0, padding: Literal['zeros', 'edge',
   'even', 'odd'] = 'zeros', axis: int = -1, **kwargs) ->
   tuple[ndarray, ndarray] | ndarray**

      Generates pitch predictions for the input signal.

      Computation is done in batches. This method is designed for
      batch processing of large numbers of inputs. It is not intended
      for use inside of loops that iterate over your data and process
      small numbers of inputs at a time.

      For small numbers of inputs that fit in one batch, directly use
      *__call__()* for faster execution, e.g., *model(x)*, or
      *model(x, training=False)* if you have layers such as
      *BatchNormalization* that behave differently during inference.

      :Parameters:
         *  **x** (ArrayLike) – Input signal(s). For a higher
            dimensional array, the pitch is detected along the last
            dimension.

         *  **fs** (``int``) – Input signal sampling rate in
            Samples/second. This must match model.fs.

         *  **p0** (``int`` | ``None``, default: ``None``) – The first
            element of the range of slices to calculate. If None then
            it is set to p_min, which is the smallest possible slice.

         *  **p1** (``int`` | ``None``, default: ``None``) – The end
            of the array. If None then the largest possible slice is
            used.

         *  **k_offset** (``int``, default: ``0``) – Index of first
            sample (t = 0) in x.

         *  **padding** (``Literal``[``'zeros'``, ``'edge'``,
            ``'even'``, ``'odd'``], default: ``'zeros'``) – Kind of
            values which are added, when the sliding window sticks out
            on either the lower or upper end of the input x. Zeros are
            added if the default ‘zeros’ is set. For ‘edge’ either the
            first or the last value of x is used. ‘even’ pads by
            reflecting the signal on the first or last sample and
            ‘odd’ additionally multiplies it with -1.

         *  **axis** (``int``, default: ``-1``) – The axis of *x* over
            which to run the model along. If not given, the last axis
            is used.

         *  **batch_size** – Number of samples per batch. If
            unspecified, *batch_size* will default to 32. Do not
            specify the *batch_size* if your data is in the form of
            dataset, generators, or *keras.utils.PyDataset* instances
            (since they generate batches).

         *  **verbose** – *“auto”*, 0, 1, or 2. Verbosity mode. 0 =
            silent, 1 = progress bar, 2 = single line. *“auto”*
            becomes 1 for most cases. Note that the progress bar is
            not particularly useful when logged to a file, so
            *verbose=2* is recommended when not running interactively
            (e.g. in a production environment). Defaults to *“auto”*.

         *  **steps** – Total number of steps (batches of samples)
            before declaring the prediction round finished. Ignored
            with the default value of *None*. If *x* is a
            *tf.data.Dataset* and *steps* is *None*, *predict()* will
            run until the input dataset is exhausted.

         *  **callbacks** – List of *keras.callbacks.Callback*
            instances. List of callbacks to apply during prediction.

      :Returns:
         If postprocessor is assigned:
            *  f0: predicted pitches

         If self.return_f0 is true:
            *  f0: predicted pitches

            *  confidence: pitch prediction confidences

         If self.return_f0 is false:
            *  2D array of a sequence of confidence levels of all
               frequency bins

      :Return type:
         ``tuple``[``ndarray``, ``ndarray``] | ``ndarray``
