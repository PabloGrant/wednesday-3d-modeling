import asyncio
import gc
import json
import os
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

app = FastAPI(title="Wednesday 3D Modeling — TRELLIS.2")
executor = ThreadPoolExecutor(max_workers=1)

TRELLIS2_MIN_VRAM_GB = 20.0
HF_MODEL = "microsoft/TRELLIS.2-4B"

# task_id -> {status, messages, lock, result_path, media_type, filename, error}
tasks: dict = {}


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


def _strip_webp_from_glb(glb_path: str) -> str:
    """Re-export GLB via trimesh to convert WebP textures → PNG (Blender compat)."""
    import trimesh as _trimesh
    scene = _trimesh.load(glb_path, process=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    out_path = tmp.name
    tmp.close()
    scene.export(out_path)
    return out_path


def _decimate_scene(glb_path: str, max_triangles: int) -> str:
    """Load GLB, decimate to max_triangles total faces, re-export. Returns new path."""
    import trimesh as _trimesh
    scene = _trimesh.load(glb_path, process=False)
    geoms = scene.geometry if isinstance(scene, _trimesh.Scene) else {"mesh": scene}
    total = sum(len(g.faces) for g in geoms.values() if hasattr(g, "faces"))
    if total <= max_triangles:
        return glb_path
    ratio = max_triangles / total
    if isinstance(scene, _trimesh.Scene):
        for name, geom in scene.geometry.items():
            if hasattr(geom, "faces") and len(geom.faces) > 4:
                target = max(4, int(len(geom.faces) * ratio))
                scene.geometry[name] = geom.simplify_quadric_decimation(target)
    else:
        scene = scene.simplify_quadric_decimation(max(4, int(total * ratio)))
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    out_path = tmp.name
    tmp.close()
    scene.export(out_path)
    return out_path


def _progress(task_id: str, msg: str):
    task = tasks.get(task_id)
    if task:
        with task["lock"]:
            task["messages"].append(msg)


def _run_generation(task_id: str, img_path: str, fmt: str, original_filename: str, decimation_target: int = 500000):
    """Blocking generation — runs in thread pool executor. Loads and unloads model each call."""
    task = tasks[task_id]
    pipeline = None
    glb_path = None

    try:
        _progress(task_id, "Checking VRAM...")
        free_gb = _free_vram_gb()
        if free_gb is not None and free_gb < TRELLIS2_MIN_VRAM_GB:
            raise RuntimeError(
                f"Insufficient VRAM ({free_gb:.1f} GB free, need {TRELLIS2_MIN_VRAM_GB} GB)."
            )

        _progress(task_id, f"VRAM OK ({free_gb:.1f} GB free). Clearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()

        _progress(task_id, "Importing TRELLIS.2 pipeline...")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel

        _progress(task_id, f"Loading {HF_MODEL} (first run downloads ~15 GB, subsequent runs load from cache)...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                hf_token = Path("/run/hf_token").read_text().strip() or None
            except Exception:
                pass
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(HF_MODEL)
        pipeline.cuda()

        _progress(task_id, "Model loaded. Preprocessing image...")
        img = Image.open(img_path).convert("RGBA")
        img = pipeline.preprocess_image(img)

        _progress(task_id, "Running 3D generation — Stage 1: Sparse structure...")
        outputs, latents = pipeline.run(
            img,
            seed=1,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            shape_slat_sampler_params={
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            tex_slat_sampler_params={
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            pipeline_type="1024_cascade",
            return_latent=True,
        )

        _progress(task_id, "Stage 2: Decoding mesh with PBR materials...")
        shape_slat, tex_slat, res = latents
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

        _progress(task_id, "Extracting GLB...")
        glb_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        glb_path = glb_tmp.name
        glb_tmp.close()

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=2048,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
        glb.export(glb_path, extension_webp=True)

    finally:
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        _cleanup(img_path)

    # Hard post-process decimation: o_voxel's decimation_target is a hint and
    # the remesh step can overshoot it. Enforce the cap on the exported GLB.
    if decimation_target < 100000:
        _progress(task_id, f"Enforcing ≤{decimation_target} triangle cap on exported mesh...")
        capped = _decimate_scene(glb_path, decimation_target)
        if capped != glb_path:
            _cleanup(glb_path)
            glb_path = capped

    # Format conversion (CPU-only from here)
    stem = Path(original_filename).stem

    if fmt == "glb":
        out_path = glb_path
        media_type = "model/gltf-binary"
        filename = f"{stem}.glb"

    elif fmt == "obj":
        _progress(task_id, "Converting to OBJ (note: PBR materials will not be preserved)...")
        import trimesh as _trimesh
        mesh_obj = _trimesh.load(glb_path)
        obj_tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        out_path = obj_tmp.name
        obj_tmp.close()
        mesh_obj.export(out_path)
        _cleanup(glb_path)
        media_type = "text/plain"
        filename = f"{stem}.obj"

    elif fmt == "fbx":
        _progress(task_id, "Converting to FBX via Blender... (this can take 1-2 minutes, connection is kept alive)")
        fbx_tmp = tempfile.NamedTemporaryFile(suffix=".fbx", delete=False)
        out_path = fbx_tmp.name
        fbx_tmp.close()
        _progress(task_id, "Preparing GLB for Blender (converting WebP textures)...")
        blender_glb = _strip_webp_from_glb(glb_path)
        blender_script = "\n".join([
            "import bpy",
            "bpy.ops.wm.read_factory_settings(use_empty=True)",
            f"bpy.ops.import_scene.gltf(filepath=r'{blender_glb}')",
            f"bpy.ops.export_scene.fbx(filepath=r'{out_path}', use_selection=False, path_mode='COPY', embed_textures=True)",
        ])
        script_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script_path = script_tmp.name
        script_tmp.write(blender_script)
        script_tmp.close()
        result = subprocess.run(
            ["blender", "--background", "--python", script_path],
            capture_output=True, text=True, timeout=120,
        )
        _cleanup(glb_path, blender_glb, script_path)
        if result.returncode != 0:
            raise RuntimeError(f"Blender FBX conversion failed: {result.stderr[-400:]}")
        media_type = "application/octet-stream"
        filename = f"{stem}.fbx"

    else:
        raise RuntimeError(f"Unknown format: {fmt}")

    _progress(task_id, f"Done! {filename} is ready.")
    with task["lock"]:
        task["status"] = "done"
        task["result_path"] = out_path
        task["media_type"] = media_type
        task["filename"] = filename


def _run_generation_safe(task_id: str, img_path: str, fmt: str, original_filename: str, decimation_target: int = 500000):
    try:
        _run_generation(task_id, img_path, fmt, original_filename, decimation_target)
    except Exception as e:
        task = tasks.get(task_id)
        if task:
            _progress(task_id, f"ERROR: {e}")
            with task["lock"]:
                task["status"] = "error"
                task["error"] = str(e)


def _run_conversion(task_id: str, glb_path: str, fmt: str, original_filename: str, max_triangles: int = 0):
    """Convert an existing GLB to obj or fbx. No GPU needed."""
    task = tasks[task_id]
    out_path = None
    try:
        stem = Path(original_filename).stem

        # Optional decimation before conversion
        if max_triangles > 0:
            _progress(task_id, f"Decimating mesh to ≤{max_triangles} triangles...")
            decimated = _decimate_scene(glb_path, max_triangles)
            if decimated != glb_path:
                _cleanup(glb_path)
                glb_path = decimated

        if fmt == "obj":
            _progress(task_id, "Converting GLB → OBJ (note: PBR materials will not be preserved)...")
            import trimesh as _trimesh
            mesh_obj = _trimesh.load(glb_path)
            obj_tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
            out_path = obj_tmp.name
            obj_tmp.close()
            mesh_obj.export(out_path)
            media_type = "text/plain"
            filename = f"{stem}.obj"

        elif fmt == "fbx":
            _progress(task_id, "Converting GLB → FBX via Blender... (this can take 1-2 minutes, connection is kept alive)")
            fbx_tmp = tempfile.NamedTemporaryFile(suffix=".fbx", delete=False)
            out_path = fbx_tmp.name
            fbx_tmp.close()
            _progress(task_id, "Preparing GLB for Blender (converting WebP textures)...")
            blender_glb = _strip_webp_from_glb(glb_path)
            blender_script = "\n".join([
                "import bpy",
                "bpy.ops.wm.read_factory_settings(use_empty=True)",
                f"bpy.ops.import_scene.gltf(filepath=r'{blender_glb}')",
                f"bpy.ops.export_scene.fbx(filepath=r'{out_path}', use_selection=False, path_mode='COPY', embed_textures=True)",
            ])
            script_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            script_path = script_tmp.name
            script_tmp.write(blender_script)
            script_tmp.close()
            result = subprocess.run(
                ["blender", "--background", "--python", script_path],
                capture_output=True, text=True, timeout=120,
            )
            _cleanup(blender_glb, script_path)
            if result.returncode != 0:
                raise RuntimeError(f"Blender FBX conversion failed: {result.stderr[-400:]}")
            media_type = "application/octet-stream"
            filename = f"{stem}.fbx"

        else:
            raise RuntimeError(f"Unknown format: {fmt}")

        _progress(task_id, f"Done! {filename} is ready.")
        with task["lock"]:
            task["status"] = "done"
            task["result_path"] = out_path
            task["media_type"] = media_type
            task["filename"] = filename

    except Exception as e:
        if out_path:
            _cleanup(out_path)
        _progress(task_id, f"ERROR: {e}")
        with task["lock"]:
            task["status"] = "error"
            task["error"] = str(e)
    finally:
        _cleanup(glb_path)


def _run_conversion_safe(task_id: str, glb_path: str, fmt: str, original_filename: str, max_triangles: int = 0):
    try:
        _run_conversion(task_id, glb_path, fmt, original_filename, max_triangles)
    except Exception as e:
        task = tasks.get(task_id)
        if task:
            _progress(task_id, f"ERROR: {e}")
            with task["lock"]:
                task["status"] = "error"
                task["error"] = str(e)


@app.get("/health")
def health():
    free = _free_vram_gb()
    return {
        "status": "ok",
        "model": HF_MODEL,
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
    image: UploadFile = File(...),
    format: str = Form(default="glb"),
    decimation_target: int = Form(default=500000),
):
    if format not in ("glb", "obj", "fbx"):
        raise HTTPException(400, "format must be glb, obj, or fbx")

    img_bytes = await image.read()
    img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img_tmp.write(img_bytes)
    img_tmp.close()

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "messages": [],
        "lock": threading.Lock(),
        "result_path": None,
        "media_type": None,
        "filename": None,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_generation_safe,
        task_id, img_tmp.name, format, image.filename or "model.png", decimation_target
    )

    return {"task_id": task_id}


@app.post("/convert")
async def convert_glb(
    glb: UploadFile = File(...),
    format: str = Form(default="obj"),
    max_triangles: int = Form(default=0),
):
    if format not in ("obj", "fbx"):
        raise HTTPException(400, "format must be obj or fbx")

    glb_bytes = await glb.read()
    glb_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    glb_tmp.write(glb_bytes)
    glb_tmp.close()

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "messages": [],
        "lock": threading.Lock(),
        "result_path": None,
        "media_type": None,
        "filename": None,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_conversion_safe,
        task_id, glb_tmp.name, format, glb.filename or "model.glb", max_triangles
    )

    return {"task_id": task_id}


@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]

    async def event_gen():
        sent = 0
        idle_ticks = 0
        while True:
            with task["lock"]:
                new_msgs = task["messages"][sent:]
                status = task["status"]

            if new_msgs:
                idle_ticks = 0
                for msg in new_msgs:
                    sent += 1
                    yield f"data: {json.dumps({'type': 'progress', 'msg': msg})}\n\n"
            else:
                idle_ticks += 1
                # Send SSE comment keepalive every ~5s to prevent Traefik from
                # closing the idle connection during long operations (e.g. Blender FBX)
                if idle_ticks % 13 == 0:
                    yield ": keepalive\n\n"

            if status == "done":
                with task["lock"]:
                    fn = task["filename"]
                yield f"data: {json.dumps({'type': 'done', 'filename': fn})}\n\n"
                break
            elif status == "error":
                with task["lock"]:
                    err = task["error"]
                yield f"data: {json.dumps({'type': 'error', 'msg': err})}\n\n"
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    with task["lock"]:
        status = task["status"]
        path = task["result_path"]
        media_type = task["media_type"]
        filename = task["filename"]
    if status != "done" or not path:
        raise HTTPException(404, "Result not ready")

    async def cleanup_task():
        await asyncio.sleep(300)
        _cleanup(path)
        tasks.pop(task_id, None)

    asyncio.create_task(cleanup_task())

    return FileResponse(path, media_type=media_type, filename=filename)
