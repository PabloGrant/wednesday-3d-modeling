FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TORCH_HOME=/models/torch
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl build-essential ninja-build \
    blender \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 -> python so scripts work
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# TRELLIS core dependencies
RUN pip install --no-cache-dir \
    easydict \
    omegaconf \
    "imageio[ffmpeg]" \
    tqdm \
    einops \
    plyfile \
    "trimesh[all]" \
    xatlas \
    "transformers>=4.40.0" \
    accelerate \
    diffusers \
    huggingface_hub \
    spconv-cu120

# nvdiffrast requires CUDA_HOME and --no-build-isolation to see PyTorch
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast

# Install TRELLIS from source (basic install)
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /opt/trellis && \
    cd /opt/trellis && pip install --no-cache-dir -e ".[basic]"

# FastAPI service
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    Pillow

WORKDIR /app
COPY app/ /app/

EXPOSE 5309

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5309", \
     "--timeout-keep-alive", "600", "--workers", "1"]
