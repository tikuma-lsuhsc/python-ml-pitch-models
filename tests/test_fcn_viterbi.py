import numpy as np
from matplotlib import pyplot as plt
import ffmpegio as ff
from pickle import load, dump
from os import path

from ml_pitch_models import pretrained as models

Model = models.FCN993Model

fs, x = ff.audio.read(
    "examples/the_north_wind_and_the_sun.wav", ar=Model.fs, ac=1, sample_fmt="flt"
)
x = x[:, 0]

# hop = "native"
hop = None

model = Model(hop=hop, postprocessor='viterbi')
f2 = model.predict(x, fs)

model = Model(return_f0=False, hop=hop)
conf = model.predict(x, fs)

model = Model(hop=hop, postprocessor='hmm')
f1 = model.predict(x, fs)

t = model.t(len(x))

plt.subplots(2,1,sharex=True)
plt.subplot(2,1,1)
plt.imshow(conf.T, origin="lower", aspect="auto", extent=[0,t[-1],model.f[0],model.f[-1]])
plt.subplot(2,1,2)
plt.plot(model.f[np.argmax(conf,axis=1)],'mx-')
plt.plot(f1, "w.-")
plt.plot(f2, "y.-")
# plt.plot(observations, "y.-")
plt.show()
