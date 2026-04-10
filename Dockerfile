FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TORCH_HOME=/models/torch
ENV CUDA_HOME=/usr/local/cuda
# sm_120 = RTX PRO 6000 Blackwell Workstation Edition
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
# Use PyTorch built-in SDPA — flash_attn has no sm_120 support
ENV ATTN_BACKEND=sdpa

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl build-essential ninja-build \
    libjpeg-dev \
    blender \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && pip install --upgrade pip

# PyTorch with CUDA 12.8
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Core runtime deps (mirrors setup.sh --basic)
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless \
    "trimesh[all]" xatlas \
    "transformers>=4.40.0" accelerate diffusers \
    huggingface_hub \
    kornia timm \
    scipy lpips zstandard pandas \
    Pillow rembg onnxruntime pygltflib \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# nvdiffrast v0.4.0 (compiled against installed torch)
# pip install places only the .so C extension; we must also copy the Python package
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast --no-build-isolation && \
    cp -r /tmp/nvdiffrast/nvdiffrast /usr/local/lib/python3.10/dist-packages/ && \
    sed -i 's/__version__ = version.*$/__version__ = "0.4.0"/' \
        /usr/local/lib/python3.10/dist-packages/nvdiffrast/__init__.py

# nvdiffrec renderutils branch (PBR rendering for TRELLIS.2)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec && \
    pip install /tmp/nvdiffrec --no-build-isolation

# CuMesh
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && \
    pip install /tmp/CuMesh --no-build-isolation

# FlexGEMM
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && \
    pip install /tmp/FlexGEMM --no-build-isolation

# TRELLIS.2 repo + submodules
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git /opt/trellis2 && \
    cd /opt/trellis2 && git submodule update --init --recursive --depth 1

# o-voxel (bundled in the TRELLIS.2 repo)
RUN pip install /opt/trellis2/o-voxel --no-build-isolation

# Patch TRELLIS.2 attention modules: replace flash_attn default with sdpa.
# flash_attn v2 does not support sm_120 (Blackwell). PyTorch sdpa does.
COPY patch_trellis2_attn.py /tmp/
RUN python3 /tmp/patch_trellis2_attn.py

# FastAPI service
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

ENV PYTHONPATH=/opt/trellis2

WORKDIR /app
COPY app/ /app/

EXPOSE 5309

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5309", \
     "--timeout-keep-alive", "600", "--workers", "1"]
