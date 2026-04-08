FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl build-essential ninja-build \
    blender \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# TRELLIS core dependencies
RUN pip3 install --no-cache-dir \
    easydict \
    omegaconf \
    imageio \
    imageio-ffmpeg \
    tqdm \
    einops \
    plyfile \
    "trimesh[all]" \
    xatlas \
    nvdiffrast \
    transformers>=4.40.0 \
    accelerate \
    diffusers \
    huggingface_hub \
    spconv-cu120

# Install TRELLIS from source
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /opt/trellis && \
    cd /opt/trellis && pip3 install --no-cache-dir -e ".[basic]"

# FastAPI service
RUN pip3 install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    Pillow

WORKDIR /app
COPY app/ /app/

EXPOSE 5309

# Long timeout for TRELLIS inference (can take several minutes)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5309", \
     "--timeout-keep-alive", "600", "--workers", "1"]
