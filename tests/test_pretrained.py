from scipy.io import wavfile
import numpy as np

from matplotlib import pyplot as plt

from fcn_f0.pretrained import *

print(available_models())

# fs, x = wavfile.read("tests/test_crepe.wav")
fs, x = wavfile.read("tests/test.wav")
# x = x[10000:20000]
x = x[: len(x) // 2]

t, f0, conf = predict(x, fs, voice_threshold=0.5,framewise=True)

plt.subplots(3, 1, sharex=True)
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(x)) / fs, x)
plt.subplot(3, 1, 2)
plt.plot(t, f0, ".-")
plt.subplot(3, 1, 3)
plt.plot(t, conf, ".-")
plt.show()
