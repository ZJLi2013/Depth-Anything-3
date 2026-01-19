# syntax=docker/dockerfile:1
# DA3 (Depth-Anything-3) CUDA 12.8 runtime + build toolchain
#
# Build:
#   docker build -t da3:latest .
#
# Run (example):
#   docker run --gpus all --rm -it -v /dc/david:/dataset -v /home/david/3d:/workspace da3:latest
#
# Notes:
# - Base image already includes PyTorch 2.9.0 + CUDA 12.8 runtime.
# - We install CUDA *toolkit* (nvcc) for extensions like gsplat.
# - This Dockerfile expects the Depth-Anything-3 repo to be in the build context root.

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit 12.8 (nvcc etc.) for compiling CUDA extensions
# (base image provides runtime only)
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# Copy project
COPY . /workspace

# Python deps
# - Do NOT let pip replace the preinstalled torch in the base image.
# - xformers 0.0.33 is known to work with your setup notes.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --no-deps xformers==0.0.33 && \
    python -m pip install --no-cache-dir torchvision && \
    python -m pip install --no-cache-dir -e .


    
# gsplat (CUDA extension) pinned commit used in notes
RUN python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"

# Default to interactive shell
CMD ["/bin/bash"]
