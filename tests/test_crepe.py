import numpy as np
from ml_pitch_models import pretrained as models
from matplotlib import pyplot as plt
import ffmpegio as ff

fs, x = ff.audio.read(
    "examples/the_north_wind_and_the_sun.wav", ar=16000, ac=1, sample_fmt="flt"
)
x = x[:, 0]

hop = None

Model = models.CrepeSmallModel

model = Model(return_f0=True)

t = model.t(len(x))

f0, conf = model.predict(x, fs)
print(f0.shape)

plt.subplots(2, 1, sharex=True)
plt.subplot(2, 1, 1)
plt.plot(t, f0)
plt.subplot(2, 1, 2)
plt.plot(t, conf)

model = Model(hop=hop)

print(model.p_range(len(x)))
f = model.f
t = model.t(len(x))

C = model.predict(x, fs)
print(C.shape)

plt.figure()
plt.pcolormesh(t, f, C.T)
# plt.show()

plt.show()
