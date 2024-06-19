import ffmpegio as ff
import numpy as np

from scipy.signal import ShortTimeFFT, get_window
from matplotlib import pyplot as plt


audiofile = "examples/the_north_wind_and_the_sun.wav"
data = {
    fs: x[:, 0]
    for fs, x in (
        ff.audio.read(audiofile, ac=1, ar=fs, sample_fmt="flt") for fs in (8000, 16000)
    )
}

fs = 8000
x = data[fs]

nb_samples = len(x)
t = np.arange(nb_samples) / fs


nperseg = int(0.05 * fs)
SFT = ShortTimeFFT(win=get_window("hamming", nperseg), hop=nperseg // 2, fs=fs)
Sxx_dB = 10 * np.log10(SFT.spectrogram(x))

import ml_pitch_models as mlf0

models = {
    model: mlf0.load_model(model, return_f0=True) for model in mlf0.available_models()
}

results = {
    name: (model.t(len(data[model.fs])), *model.predict(data[model.fs], model.fs))
    for name, model in models.items()
}

for model, res in results.items():
    plt.plot(*res[:2], ".-", label=model)
    plt.legend()
plt.show()

# print(results)

# plt.subplots(2, 1, sharex=True)
# plt.subplot(2, 1, 1)
# plt.imshow(Sxx_dB, extent=SFT.extent(nb_samples), aspect="auto", origin="lower")
# for model, res in results.items():
#     plt.plot(*res[:-2], ".-w", label=model)
# plt.ylabel("frequency (Hz)")
# plt.ylim([0, 1000])
# plt.legend()
# plt.subplot(2, 1, 2)
# for model, res in results.items():
#     plt.plot(*res[:-2], ".-w", label=model)
# plt.ylabel("confidence")
# plt.xlabel("time (s)")
