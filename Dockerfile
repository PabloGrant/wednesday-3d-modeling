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
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast --no-build-isolation

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
RUN python3 - <<'PYEOF'
import os, glob

BASE = '/opt/trellis2/trellis2/modules'

def sed(path, old, new):
    with open(path) as f: c = f.read()
    if old not in c:
        return False
    with open(path, 'w') as f: f.write(c.replace(old, new))
    return True

# 1. Dense attention: change default BACKEND
p = f'{BASE}/attention/__init__.py'
if os.path.exists(p):
    ok = sed(p, "BACKEND = 'flash_attn'", "BACKEND = 'sdpa'")
    print(f"dense attn default: {'ok' if ok else 'not found'}")

# 2. Sparse attention: change default ATTN + allow 'sdpa' in validation
p = f'{BASE}/sparse/__init__.py'
if os.path.exists(p):
    ok1 = sed(p, "ATTN = 'flash_attn'", "ATTN = 'sdpa'")
    ok2 = sed(p,
        "env_sparse_attn in ['xformers', 'flash_attn']",
        "env_sparse_attn in ['xformers', 'flash_attn', 'sdpa']")
    print(f"sparse attn default: {'ok' if ok1 else 'not found'}, validation: {'ok' if ok2 else 'not found'}")

# 3. Patch import blocks + add sdpa compute in sparse attention files
IMPORT_OLD = """if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")"""

IMPORT_NEW = """if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
elif ATTN == 'sdpa':
    import torch.nn.functional as _F_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")"""

# Uniform window branch (same in serialized + windowed)
UNIFORM_OLD = """        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)                       # [B, N, H, C]
            out = xops.memory_efficient_attention(q, k, v)          # [B, N, H, C]
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
        out = out.reshape(B * N, H, C)                              # [M, H, C]"""

UNIFORM_NEW = """        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)                       # [B, N, H, C]
            out = xops.memory_efficient_attention(q, k, v)          # [B, N, H, C]
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]
        elif ATTN == 'sdpa':
            q, k, v = qkv_feats.unbind(dim=2)                       # [B, N, H, C]
            q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3); v = v.permute(0, 2, 1, 3)
            out = _F_attn.scaled_dot_product_attention(q, k, v)     # [B, H, N, C]
            out = out.permute(0, 2, 1, 3)                           # [B, N, H, C]
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
        out = out.reshape(B * N, H, C)                              # [M, H, C]"""

# Varlen branch (same in serialized + windowed)
VARLEN_OLD = """        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \\
                        .to(qkv.device).int()
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]"""

VARLEN_NEW = """        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \\
                        .to(qkv.device).int()
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]
        elif ATTN == 'sdpa':
            out_parts = []
            start = 0
            for sl in seq_lens:
                qi, ki, vi = qkv_feats[start:start+sl].unbind(dim=1)
                qi = qi.unsqueeze(0).permute(0, 2, 1, 3)
                ki = ki.unsqueeze(0).permute(0, 2, 1, 3)
                vi = vi.unsqueeze(0).permute(0, 2, 1, 3)
                oi = _F_attn.scaled_dot_product_attention(qi, ki, vi)
                out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
                start += sl
            out = torch.cat(out_parts, dim=0)                       # [M, H, C]"""

# full_attn varlen compute
FULL_OLD = """    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")"""

FULL_NEW = """    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif ATTN == 'sdpa':
        if num_all_args == 1:
            out_parts = []
            start = 0
            for sl in q_seqlen:
                qi, ki, vi = qkv[start:start+sl].unbind(dim=1)
                qi = qi.unsqueeze(0).permute(0, 2, 1, 3)
                ki = ki.unsqueeze(0).permute(0, 2, 1, 3)
                vi = vi.unsqueeze(0).permute(0, 2, 1, 3)
                oi = _F_attn.scaled_dot_product_attention(qi, ki, vi)
                out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
                start += sl
            out = torch.cat(out_parts, dim=0)
        elif num_all_args == 2:
            out_parts = []
            qs, kvs = 0, 0
            for q_sl, kv_sl in zip(q_seqlen, kv_seqlen):
                qi = q[qs:qs+q_sl].unsqueeze(0).permute(0, 2, 1, 3)
                ki, vi = kv[kvs:kvs+kv_sl].unbind(dim=1)
                ki = ki.unsqueeze(0).permute(0, 2, 1, 3)
                vi = vi.unsqueeze(0).permute(0, 2, 1, 3)
                oi = _F_attn.scaled_dot_product_attention(qi, ki, vi)
                out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
                qs += q_sl; kvs += kv_sl
            out = torch.cat(out_parts, dim=0)
        elif num_all_args == 3:
            out_parts = []
            start = 0
            for sl in q_seqlen:
                qi = q[start:start+sl].unsqueeze(0).permute(0, 2, 1, 3)
                ki = k[start:start+sl].unsqueeze(0).permute(0, 2, 1, 3)
                vi = v[start:start+sl].unsqueeze(0).permute(0, 2, 1, 3)
                oi = _F_attn.scaled_dot_product_attention(qi, ki, vi)
                out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
                start += sl
            out = torch.cat(out_parts, dim=0)
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")"""

attn_dir = f'{BASE}/sparse/attention'
for fname in ['full_attn.py', 'serialized_attn.py', 'windowed_attn.py']:
    p = f'{attn_dir}/{fname}'
    if not os.path.exists(p):
        print(f"  SKIP (not found): {p}")
        continue
    ok1 = sed(p, IMPORT_OLD, IMPORT_NEW)
    print(f"  {fname} import: {'ok' if ok1 else 'not found'}")
    if fname == 'full_attn.py':
        ok2 = sed(p, FULL_OLD, FULL_NEW)
        print(f"  {fname} compute: {'ok' if ok2 else 'not found'}")
    else:
        ok2 = sed(p, UNIFORM_OLD, UNIFORM_NEW)
        ok3 = sed(p, VARLEN_OLD, VARLEN_NEW)
        print(f"  {fname} uniform: {'ok' if ok2 else 'not found'}, varlen: {'ok' if ok3 else 'not found'}")

# Clear pyc caches
for pyc in glob.glob(f'{BASE}/**/__pycache__/*.pyc', recursive=True):
    os.unlink(pyc)
print("Done.")
PYEOF

# FastAPI service
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

ENV PYTHONPATH=/opt/trellis2

WORKDIR /app
COPY app/ /app/

EXPOSE 5309

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5309", \
     "--timeout-keep-alive", "600", "--workers", "1"]
