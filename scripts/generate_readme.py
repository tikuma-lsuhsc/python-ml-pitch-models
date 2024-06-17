# for details see https://stackoverflow.com/questions/36237477/python-docstrings-to-github-readme-md
# further details in https://blog.matteoferla.com/2019/11/convert-python-docstrings-to-github.html

import re
import shutil
import os
from os import path
import subprocess

import conf_cuda

# ## Settings
module_name = "fcn_f0"
author_name = "Takeshi Ikuma (LSUHSC)"

src_dir = "readme_src"
build_dir = "_build"

# ## Apidoc call
arguments = [
    "-b",
    "rst",
    "-v",
    "--keep-going",
    src_dir,
    build_dir,
]
proc = subprocess.run(["sphinx-build", *arguments], capture_output=True)
if proc.stderr:
    raise RuntimeError(proc.stderr.decode())
print(proc.stdout.decode())

shutil.copyfile(path.join(build_dir, "index.rst"), "README.rst")
shutil.rmtree(build_dir, ignore_errors=True)
