
:code:`ml-pitch-models` - A collection of ML-based pitch detector Python Tensorflow models
==========================================================================================

Description
-----------

This Python package features two deep-learning pitch detection models. 

1. :code:`CrepeModel` - CREPE: deep convolutional neural network (CNN) model (5 pretrained weights)

   J. W. Kim, J. Salamon, P. Li, J. P. Bello. "CREPE: A Convolutional Representation for Pitch 
   Estimation", Proc. IEEE ICASSP, 2018. doi: 10.1109/ICASSP.2018.8461329

   GitHub Repository: https://github.com/marl/crepe

2. :code:`FcnF0Model` - FCN-F0: fully convolutional network (FCN) model (3 pretrained weights)

   L. Ardaillon and A. Roebel, "Fully-Convolutional Network for Pitch Estimation of Speech Signals", 
   Proc. Interspeech, 2019. doi: 10.21437/Interspeech.2019-2815

   GitHub Repository: https://github.com/ardaillon/FCN-f0


While both of these models are open-sourced by the respective authors, including their pre-trained 
weights, this package distributes only the models with a convenient mechanism to also distribute 
pretrained weights, minimizing the footprint of the package.


Installation
------------

To install just the models *without* any pretrained weights:

.. code-block:: bash

   pip install ml-pitch-models

Each set of pretrained weights is available as a separate PyPI package, and they can be installed
using `pip`'s the optional extra dependency specifiers. 

Based on Ardaillon and Roebel's choice, the (current) default model for `ml_pitch_models.predict()` is the 
`FCN-993` model, and to include the pretrained weights of the default model:

.. code-block:: bash
   
   pip install ml-pitch-models[DEFAULT_WEIGHTS]

To install individual pretrained weights, use the following extra dependency keywords or a
combination thereof.

.. code-block:: bash

   pip install ml-pitch-models[CREPE_FULL_WEIGHTS]
   pip install ml-pitch-models[CREPE_LARGE_WEIGHTS]
   pip install ml-pitch-models[CREPE_MEDIUM_WEIGHTS]
   pip install ml-pitch-models[CREPE_SMALL_WEIGHTS]
   pip install ml-pitch-models[CREPE_TINY_WEIGHTS]
   pip install ml-pitch-models[FCN_1953_WEIGHTS]
   pip install ml-pitch-models[FCN_993_WEIGHTS]
   pip install ml-pitch-models[FCN_929_WEIGHTS]

Tensorflow Installation
+++++++++++++++++++++++

This package only only installs the base `tensorflow` package as its dependency. To enable GPU 
acceleration (in Linux), also run:

.. code-block:: bash

   pip install tensorflow[and-cuda]

Read https://www.tensorflow.org/install/pip for more details. Users are responsible to enable the 
GPU support in tensorflow before running the pitch prediction.


Basic Usage
-----------

Assume :code:`x` is an 1D Numpy array, containing a signal sampled at 8 kHz (:code:`fs=8000`)

.. code-block:: python

   import ml_pitch_models

   t, f0, conf = ml_pitch_models.predict(fs, x)

computes the pitch estimates :code:`f0` with the default model :code:`fcn_993`. The estimates are 
generated with a sliding window of size 993 An :code:`f0` value of zero indicates that no pitch was 
detected in that window.

The timestamps :code:`t` is a sequence of the timestamps of the middle of the window. By default, 
:code:`t[0]=0` and :code:`t[-1]` is the last window with more than half of its bins filled.

The last output :code:`conf` is the confidence levels of the estimates. This is the maximum value of
the DL classifier layer if pitch is detected or :code:`1-max` if no pitch is detected.

The third argument of :code:`ml_pitch_models.predict` is the DL model to use. To find the available 
pretrained models, run

.. code-block:: python

   ml_pitch_models.available_models() # returns a list of model names

Possible model names are: :code:`"crepe_full"`, :code:`"crepe_large"`, :code:`"crepe_medium"`, 
:code:`"crepe_small"`, :code:`"crepe_tiny"`, :code:`"fcn_1953"`, :code:`"fcn_929"`, and 
:code:`"fcn_993"`. Only the installed pretrained models would be listed.

Note that the pretrained CREPE model and the 8-kHz signal :code:`x` is not compatible because the
CREPE models were pretrained with 16-kHz signals. In other words,

.. code-block:: python

   t, f0, conf = ml_pitch_models.predict(fs, x, 'crepe_large')

will throw a :code:`ValueError` exception because the signal sampling rate :code:`fs` does not match
the model's input sampling rate (16 kHz). The signal must first be interpolated by 2 to 16 kHz to
run :code:`x` through a CREPE pretrained model. For example, you can use `scipy.signal.resample_poly 
<https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly>`.


:code:`framewise` boolean option
++++++++++++++++++++++++++++++++

The key difference between the CREPE and FCN-F0 models is that the latter can run the pitch 
detection more efficiently with a window hop size that is imposed by the model. The boolean 
:code:`framewise` controls this mode of operation (default to the continuous operation):

.. code-block:: python

   t_cont, f0_cont, conf_cont = ml_pitch_models.predict(fs, x, 'fcn_929', framewise=False) # default for FCN-F0
   t_batch, f0_batch, conf_batch = ml_pitch_models.predict(fs, x, 'fcn_929', framewise=True, hop=400) # 50-ms hop size

The batch mode (:code:`framewise=True`, which is the only mode CREPE model opearates in) allows a 
model-independent hop size (:code:`hop` argument, default to a 10-ms interval in samples).

Simultaneous processing of multiple signals
+++++++++++++++++++++++++++++++++++++++++++

Both models supports processing multiple signals (of the same length) at once. The signal arrays must
be stack in "rows". Suppose that we have 3 1D signals: :code:`x0``, :code:`x1`, and :code:`x2`:

.. code-block:: python

   import numpy as np

   t, f0, conf = ml_pitch_models.predict(fs, np.stack([x0, x1, x2], axis=0))

Then, :code:`f0` and :code:`conf` are 2D arrays with 3 rows, corresponding to the input signals.

Assuming that the system is not memory constrained, this yields in a faster execution as the model is
only constructed once.

API Reference
-------------

.. autoclass:: ml_pitch_models.predict

.. autoclass:: ml_pitch_models.CrepeModel
   :members: bin_frequencies, predict

.. autoclass:: ml_pitch_models.FcnF0Model
   :members: bin_frequencies, native_hop, predict
