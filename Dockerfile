FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ninja-build \
    blender \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.4 runtime bundled in the wheel — no CUDA base image needed
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# TRELLIS core dependencies (nvdiffrast has no PyPI wheel — install from source)
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

RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast

# Install TRELLIS from source (basic install — no heavy optional CUDA extensions)
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
