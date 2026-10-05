#!/bin/bash

pip install ./src/third_party/diff-bbsplat-rasterization
pip uninstall pytorch3d
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+'https://github.com/otaheri/MANO'
pip install --no-deps --force-reinstall git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da
pip install git+https://github.com/tinyobjloader/tinyobjloader.git

