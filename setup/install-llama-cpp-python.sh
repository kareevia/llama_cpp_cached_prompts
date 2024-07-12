#!/bin/bash

sudo apt-get install libopenblas-dev
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python --no-cache-dir
