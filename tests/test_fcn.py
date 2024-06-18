import numpy as np
from ml_pitch_models import pretrained as models, utils
from matplotlib import pyplot as plt
import ffmpegio as ff

fs, x = ff.audio.read(
    "examples/the_north_wind_and_the_sun.wav", ar=8000, ac=1, sample_fmt="flt"
)
x = x[:, 0]

# model = models.FCN1953Model()
model = models.FCN929Model()
# model = models.FCN993Model(framewise=True, return_f0=True)

f, t, C = model.predict(x, fs)

plt.pcolormesh(t, f, C.T)
plt.show()
