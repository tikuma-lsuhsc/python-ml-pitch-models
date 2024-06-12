# from fcn.prediction import (
#     load_model,
#     get_infos_from_tag,
#     predict_frameWise,
#     predict_fullConv,
#     sliding_norm,
# )

from scipy.io import wavfile
import numpy as np

# def process_file(
fs, x = wavfile.read("tests/test_crepe.wav")
# fs, x = wavfile.read("tests/test.wav")

# import os
# import nvidia.cudnn

# os.environ["CUDNN_PATH"] = cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
# os.environ["LD_LIBRARY_PATH"] = (
#     os.path.join(cudnn_path, "lib") + os.pathsep + os.environ["LD_LIBRARY_PATH"]
# )
# # # print(os.environ['LD_LIBRARY_PATH'])

# print(os.environ["CUDNN_PATH"])
# print(os.environ["LD_LIBRARY_PATH"])

# export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

# exit()

# export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

# export CUDNN_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn"
# export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64"

# exit()
from fcn_f0 import pretrained as models, utils

# model = models.CrepeFullModel()
# model = models.CrepeSmallModel()
# model = models.CrepeLargeModel()
# model = models.CrepeMediumModel()
model = models.CrepeTinyModel(return_f0=True)

# from tensorflow.keras.utils import plot_model

# plot_model(model, "test.svg", show_shapes=True, show_dtype=True, show_layer_names=True)

# model = models.FCN1953Model()
# model = models.FCN929Model()

# model = models.FCN993Model(framewise=True, return_f0=True)
# audio = utils.sliding_norm(x, frame_sizes=model.nb_input)
# nblks = len(audio) // model.nb_input
# audio = audio[: nblks * model.nb_input].reshape(nblks, model.nb_input)


# model = models.FCN993Model()
# audio = utils.sliding_norm(x, frame_sizes=model.nb_input).reshape(1, -1)

# print(x.shape)
out = model.predict(x, fs)
# out = model.predict(x.reshape((1, -1)), fs)
# out = model.predict(np.tile(x.reshape((1, -1)), (2, 1)), fs)
# out = model.predict(np.tile(x.reshape((1, 1, -1)), (2, 1, 1)), fs)
if model.return_f0:
    t, f0, conf = out
    print(f0.shape)
else:
    t, y = out
    print(y.shape)
print(t.shape)
