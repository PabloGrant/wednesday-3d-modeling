"""
Patch TRELLIS.2 for RTX Pro 6000 Blackwell (sm_120) compatibility:
1. Attention: replace flash_attn default with sdpa (flash_attn v2 has no sm_120 support)
2. rembg: substitute briaai/RMBG-2.0 (gated) with ZhengPeng7/BiRefNet (public upstream)
"""
import os

BASE = '/opt/trellis2/trellis2/modules/sparse'


def sed(path, old, new, required=True):
    with open(path) as f:
        c = f.read()
    if old not in c:
        if required:
            raise RuntimeError(f"Pattern not found in {path}:\n{old[:120]!r}")
        return False
    with open(path, 'w') as f:
        f.write(c.replace(old, new))
    return True


# ── 1. config.py ─────────────────────────────────────────────────────────────
p = f'{BASE}/config.py'
sed(p, "ATTN = 'flash_attn'", "ATTN = 'sdpa'")
sed(p,
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']")
print("config.py: OK")


# ── 2. full_attn.py ───────────────────────────────────────────────────────────
# Insert sdpa branch after the flash_attn_3 block, before else/raise.
p = f'{BASE}/attention/full_attn.py'

FULL_ANCHOR = """\
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")"""

FULL_REPLACE = """\
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    elif config.ATTN == 'sdpa':
        import torch.nn.functional as _F
        if num_all_args == 1:
            out_parts = []
            start = 0
            for sl in q_seqlen:
                qi, ki, vi = qkv[start:start+sl].unbind(dim=1)
                qi = qi.unsqueeze(0).permute(0, 2, 1, 3)
                ki = ki.unsqueeze(0).permute(0, 2, 1, 3)
                vi = vi.unsqueeze(0).permute(0, 2, 1, 3)
                oi = _F.scaled_dot_product_attention(qi, ki, vi)
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
                oi = _F.scaled_dot_product_attention(qi, ki, vi)
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
                oi = _F.scaled_dot_product_attention(qi, ki, vi)
                out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
                start += sl
            out = torch.cat(out_parts, dim=0)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")"""

sed(p, FULL_ANCHOR, FULL_REPLACE)
print("full_attn.py: OK")


# ── 3. windowed_attn.py ───────────────────────────────────────────────────────
# Insert sdpa branch after the flash_attn block, before `out = out[q_bwd_indices]`.
p = f'{BASE}/attention/windowed_attn.py'

WIN_ANCHOR = """\
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]

    out = out[q_bwd_indices]"""

WIN_REPLACE = """\
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]
    elif config.ATTN == 'sdpa':
        import torch.nn.functional as _F
        k_feats, v_feats = kv_feats.unbind(dim=1)  # [M, H, C] each
        out_parts = []
        qs, kvs = 0, 0
        for q_sl, kv_sl in zip(q_seq_lens, kv_seq_lens):
            qi = q_feats[qs:qs+q_sl].unsqueeze(0).permute(0, 2, 1, 3)
            ki = k_feats[kvs:kvs+kv_sl].unsqueeze(0).permute(0, 2, 1, 3)
            vi = v_feats[kvs:kvs+kv_sl].unsqueeze(0).permute(0, 2, 1, 3)
            oi = _F.scaled_dot_product_attention(qi, ki, vi)
            out_parts.append(oi.permute(0, 2, 1, 3).squeeze(0))
            qs += q_sl; kvs += kv_sl
        out = torch.cat(out_parts, dim=0)  # [M, H, C]

    out = out[q_bwd_indices]"""

sed(p, WIN_ANCHOR, WIN_REPLACE)
print("windowed_attn.py: OK")


# ── 4. Replace BiRefNet.py with rembg-based implementation ───────────────────
# briaai/RMBG-2.0 and ZhengPeng7/BiRefNet are both gated or load gated deps.
# The rembg Python package (already installed) uses ONNX/u2net — no HF gate.
BIREFNET_NEW = '''\
from typing import *
from PIL import Image


class BiRefNet:
    """
    Background removal using the rembg package (isnet-general-use ONNX).
    Drop-in replacement for briaai/RMBG-2.0 — no gated HF model required.
    """
    def __init__(self, model_name: str = "isnet-general-use", **kwargs):
        from rembg import new_session
        self._session = new_session(
            model_name if "RMBG" not in model_name and "BiRefNet" not in model_name
            else "isnet-general-use"
        )

    def to(self, device):
        return self  # rembg uses ONNX/CPU — device is a no-op

    def cpu(self):
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        from rembg import remove
        return remove(image, session=self._session)
'''

with open('/opt/trellis2/trellis2/pipelines/rembg/BiRefNet.py', 'w') as f:
    f.write(BIREFNET_NEW)
print("BiRefNet.py: OK")

print("All patches applied.")
