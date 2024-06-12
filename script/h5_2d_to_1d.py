import h5py
from os import path
import re
import numpy as np

datadir = path.join("src", "fcn_f0", "data")
h5files = [
    path.join(datadir, f)
    for f in (
        *(
            path.join("crepe", f"model-{s}.h5")
            for s in ("full", "large", "medium", "small", "tiny")
        ),
        *(path.join(s, "weights.h5") for s in ("FCN_929", "FCN_993", "FCN_1953")),
    )
]

re_layer = re.compile(r"^conv\d$|^classifier$")

for file in h5files:
    print(file)
    with h5py.File(file, "r+") as f:

        for layer in f:
            if re_layer.match(layer):
                for g in f[layer]:
                    w = f[layer][g]["kernel:0"]
                    shape = w.shape
                    if len(shape) == 4:
                        shape = shape[2:]
                        new_w = [d.reshape(shape) for d in w]
                        del f[layer][g]["kernel:0"]
                        f[layer][g]["kernel:0"] = new_w
                        print(f"{layer}/{g}", f[layer][g]["kernel:0"].shape)
