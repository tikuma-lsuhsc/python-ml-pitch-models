import sysconfig, os
from os import path

libpath = sysconfig.get_path("purelib")

cudnn_path = path.join(libpath, "nvidia", "cudnn")
if path.exists(cudnn_path):
    os.environ["CUDNN_PATH"] = cudnn_path
    os.environ["LD_LIBRARY_PATH"] = (
        os.path.join(cudnn_path, "lib") + os.pathsep + os.environ["LD_LIBRARY_PATH"]
    )

tensorrt_libs_path = path.join(libpath, "tensorrt_libs")
if path.exists(tensorrt_libs_path):
    os.environ['LD_LIBRARY_PATH'] = tensorrt_libs_path + os.pathsep + os.environ["LD_LIBRARY_PATH"]