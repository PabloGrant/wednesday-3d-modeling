import gc
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI(title="Wednesday 3D Modeling")

TRELLIS_MIN_VRAM_GB = 20.0


def _free_vram_gb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        return None


def _cleanup(*paths: str | None):
    for p in paths:
        try:
            if p and Path(p).exists():
                os.unlink(p)
        except Exception:
            pass


@app.get("/health")
def health():
    free = _free_vram_gb()
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "vram_free_gb": round(free, 1) if free is not None else None,
    }


@app.get("/vram")
def vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        name, used, free, total = [x.strip() for x in r.stdout.strip().split(",")]
        pids_r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        processes = [l.strip() for l in pids_r.stdout.strip().splitlines() if l.strip()]
        return {
            "gpu": name,
            "used_mb": int(used), "free_mb": int(free), "total_mb": int(total),
            "processes": processes,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    format: str = Form(default="glb"),
):
    if format not in ("glb", "obj", "fbx"):
        raise HTTPException(400, "format must be glb, obj, or fbx")

    # Check VRAM before attempting to load — fail fast if GPU is squatted
    free_gb = _free_vram_gb()
    if free_gb is not None and free_gb < TRELLIS_MIN_VRAM_GB:
        raise HTTPException(
            503,
            f"Insufficient VRAM ({free_gb:.1f} GB free, need {TRELLIS_MIN_VRAM_GB} GB). "
            "Another process may be occupying GPU memory. Free VRAM and retry.",
        )

    # Clear any cached tensors from previous runs in this process
    torch.cuda.empty_cache()
    gc.collect()

    # Save uploaded image
    img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img_tmp.write(await image.read())
    img_tmp.close()
    img_path = img_tmp.name

    glb_path: str | None = None
    out_path: str | None = None
    pipeline = None

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()

        img = Image.open(img_path).convert("RGBA")
        outputs = pipeline.run(img, seed=1)

        glb_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        glb_path = glb_tmp.name
        glb_tmp.close()

        glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
        glb.export(glb_path)

    finally:
        # Always unload model and clear GPU memory after generation
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    # Format conversion (CPU-only from here)
    extra_cleanup: list[str | None] = [img_path]

    if format == "glb":
        out_path = glb_path
        media_type = "model/gltf-binary"
        out_suffix = ".glb"

    elif format == "obj":
        import trimesh as _trimesh
        mesh = _trimesh.load(glb_path)
        obj_tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        out_path = obj_tmp.name
        obj_tmp.close()
        mesh.export(out_path)
        extra_cleanup.append(glb_path)
        media_type = "text/plain"
        out_suffix = ".obj"

    elif format == "fbx":
        fbx_tmp = tempfile.NamedTemporaryFile(suffix=".fbx", delete=False)
        out_path = fbx_tmp.name
        fbx_tmp.close()

        blender_script = "\n".join([
            "import bpy",
            "bpy.ops.wm.read_factory_settings(use_empty=True)",
            f"bpy.ops.import_scene.gltf(filepath=r'{glb_path}')",
            f"bpy.ops.export_scene.fbx(filepath=r'{out_path}', use_selection=False)",
        ])
        script_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script_path = script_tmp.name
        script_tmp.write(blender_script)
        script_tmp.close()
        extra_cleanup += [glb_path, script_path]

        result = subprocess.run(
            ["blender", "--background", "--python", script_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            background_tasks.add_task(_cleanup, *extra_cleanup, out_path)
            raise HTTPException(500, f"Blender FBX conversion failed: {result.stderr[-400:]}")

        media_type = "application/octet-stream"
        out_suffix = ".fbx"

    stem = Path(image.filename or "model").stem
    background_tasks.add_task(_cleanup, *extra_cleanup, out_path)

    return FileResponse(
        out_path,
        media_type=media_type,
        filename=f"{stem}{out_suffix}",
    )
