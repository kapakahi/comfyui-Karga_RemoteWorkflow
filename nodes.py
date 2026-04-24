"""
Karga Remote Workflow
=====================
Single node that:
  - Scans RemoteWorkflowNodes/workflows/ for JSON files (dropdown picker)
  - Accepts a ComfyUI IMAGE tensor as input, uploads it to the remote instance
  - Exposes a prompt text box and seed controls
  - Dynamically exposes other [ui]-tagged fields from the selected workflow
  - Queues the job on a remote ComfyUI instance and returns the output image

[ui] Tag Convention (in your remote workflow):
  Add [ui] anywhere in a node's title for automatic detection:
    "CLIP Text Encode [ui]"        -> prompt field (text)
    "KSampler [ui]"                -> seed field
    "Load Image [ui]"              -> input_image field
    "Checkpoint Loader [ui]"       -> model field

  For custom label/input override:  [ui] label_name:input_field
    "[ui] steps:steps"
    "[ui] cfg:cfg"
"""

import json
import uuid
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
from PIL import Image
import io
import time
import os
import copy
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

NODE_DIR      = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(NODE_DIR, "workflows")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_workflows() -> list:
    if not os.path.isdir(WORKFLOWS_DIR):
        os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(WORKFLOWS_DIR) if f.lower().endswith(".json"))
    return files if files else ["(no workflows found)"]


def _load_workflow(filename: str) -> dict:
    path = os.path.join(WORKFLOWS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_ui_nodes(workflow: dict) -> dict:
    """
    Scans workflow for nodes with [ui] anywhere in _meta.title.

    Simple tag:    any title containing [ui]  e.g. "CLIP Text Encode [ui]"
                                                    "[ui] My Node"
                                                    "KSampler [ui] advanced"
    Explicit override: [ui] label:input_key   e.g. "[ui] prompt:text"

    Class-type auto-detection (simple tag):
      CLIPTextEncode / CLIPTextEncodeSDXL -> label="prompt",      input_key="text" / "text_g"
      KSampler / KSamplerAdvanced         -> label="seed",        input_key="seed" / "noise_seed"
      RandomNoise                         -> label="seed",        input_key="noise_seed"
      LoadImage                           -> label="input_image", input_key="image"
      LoadImageMask                       -> label="input_mask",  input_key="image"
      CheckpointLoaderSimple              -> label="model",       input_key="ckpt_name"
      LoraLoader                          -> label="lora",        input_key="lora_name"
      (unknown)                           -> first scalar input,  class_type as label

    Returns:  { "label_name": ("node_id", "input_key"), ... }
    """
    CLASS_MAP = {
        "CLIPTextEncode":         ("prompt",      "text"),
        "CLIPTextEncodeSDXL":     ("prompt",      "text_g"),
        "KSampler":               ("seed",        "seed"),
        "KSamplerAdvanced":       ("seed",        "noise_seed"),
        "RandomNoise":            ("seed",        "noise_seed"),
        "LoadImage":              ("input_image", "image"),
        "LoadImageMask":          ("input_mask",  "image"),
        "CheckpointLoaderSimple": ("model",       "ckpt_name"),
        "LoraLoader":             ("lora",        "lora_name"),
    }

    ui_map = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        title      = node.get("_meta", {}).get("title", "")
        title_low  = title.lower()
        class_type = node.get("class_type", "")

        if "[ui]" not in title_low:
            continue

        # Explicit override: find "[ui] label:input_key" anywhere in the title
        ui_pos = title_low.find("[ui]")
        after  = title[ui_pos + 4:].strip()
        if ":" in after and not after.startswith(":"):
            label, input_key = after.split(":", 1)
            label     = label.strip()
            input_key = input_key.strip()
            if label and input_key:
                ui_map[label] = (node_id, input_key)
                continue

        # Auto-detect from class_type
        if class_type in CLASS_MAP:
            label, input_key = CLASS_MAP[class_type]
            ui_map[label] = (node_id, input_key)
            continue

        # Fallback: first scalar input, class_type as label
        input_key = next(
            (k for k, v in node.get("inputs", {}).items()
             if isinstance(v, (str, int, float))),
            None,
        )
        label = class_type.lower()
        if label and input_key:
            ui_map[label] = (node_id, input_key)

    return ui_map


def _frames_from_mp4(mp4_bytes: bytes) -> list:
    """
    Extract frames from raw mp4 bytes. Returns a list of float32 [H,W,C] numpy arrays.
    Tries imageio+imageio-ffmpeg first, falls back to cv2.
    """
    try:
        import imageio
        import imageio_ffmpeg  # noqa: F401 — ensures ffmpeg backend is available
        buf = io.BytesIO(mp4_bytes)
        reader = imageio.get_reader(buf, format="mp4")
        frames = [np.array(f).astype(np.float32) / 255.0 for f in reader]
        reader.close()
        return frames
    except ImportError:
        pass

    try:
        import cv2
        arr = np.frombuffer(mp4_bytes, dtype=np.uint8)
        cap = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        # imdecode only gets one frame; use VideoCapture via temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(mp4_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        os.unlink(tmp_path)
        return frames
    except ImportError:
        pass

    raise RuntimeError(
        "[KargaRemoteWorkflow] Cannot decode mp4 — install imageio and imageio-ffmpeg, or opencv-python:\n"
        "  pip install imageio imageio-ffmpeg\n"
        "  # or\n"
        "  pip install opencv-python"
    )


def _post_upload(remote_address: str, img_bytes: bytes, filename: str) -> str:
    """POST raw image bytes to /upload/image. Returns the filename the server assigned."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{remote_address}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        result = json.loads(res.read())
    return result.get("name", filename)


def _upload_image(remote_address: str, img_tensor, filename: str = None) -> str:
    """
    Upload a [1,H,W,C] IMAGE (torch.Tensor or np.ndarray) to /upload/image.
    Returns the filename the server assigned.
    """
    frame = img_tensor[0]
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    img_np  = (np.array(frame) * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG")

    if filename is None:
        filename = f"rwf_input_{uuid.uuid4().hex[:8]}.png"

    assigned = _post_upload(remote_address, buf.getvalue(), filename)
    print(f"[KargaRemoteWorkflow] Uploaded input image → {assigned}")
    return assigned


def _upload_mask(remote_address: str, mask_tensor) -> str:
    """
    Upload a [B,H,W] or [H,W] MASK (torch.Tensor or np.ndarray) to /upload/image.
    Returns the filename the server assigned.
    """
    # MASK tensors are [B,H,W]; take first frame to get [H,W]
    mask_frame = mask_tensor[0] if mask_tensor.ndim == 3 else mask_tensor
    if hasattr(mask_frame, "cpu"):
        mask_frame = mask_frame.cpu().numpy()
    mask_np  = (np.array(mask_frame) * 255).clip(0, 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_np, mode="L")
    buf      = io.BytesIO()
    pil_mask.save(buf, format="PNG")

    filename = f"rwf_mask_{uuid.uuid4().hex[:8]}.png"
    assigned = _post_upload(remote_address, buf.getvalue(), filename)
    print(f"[KargaRemoteWorkflow] Uploaded mask → {assigned}")
    return assigned


# ── Node ──────────────────────────────────────────────────────────────────────

# Reserved labels handled explicitly — excluded from dynamic [ui] field discovery.
# "seed" is reserved because the fixed seed input already provides the widget + randomizer.
RESERVED = {
    "workflow", "remote_address", "prompt", "seed",
    "image", "mask", "poll_interval", "timeout", "image_index",
}


class KargaRemoteWorkflow:

    @classmethod
    def INPUT_TYPES(cls):
        workflows = _list_workflows()

        # Discover extra [ui] fields from first workflow, skipping reserved names
        ui_fields = {}
        try:
            if workflows and workflows[0] != "(no workflows found)":
                wf     = _load_workflow(workflows[0])
                ui_map = _find_ui_nodes(wf)
                for label, (node_id, input_key) in ui_map.items():
                    if label in RESERVED or input_key in ("image", "seed", "noise_seed"):
                        continue
                    existing = wf[node_id]["inputs"].get(input_key)
                    if isinstance(existing, bool):
                        ui_fields[label] = ("BOOLEAN", {"default": existing})
                    elif isinstance(existing, int):
                        ui_fields[label] = ("INT",   {"default": existing, "min": -999999, "max": 999999})
                    elif isinstance(existing, float):
                        ui_fields[label] = ("FLOAT", {"default": existing, "min": -999999.0, "max": 999999.0, "step": 0.01})
                    else:
                        ui_fields[label] = ("STRING", {"default": str(existing) if existing is not None else "", "multiline": False})
        except Exception as e:
            print(f"[KargaRemoteWorkflow] WARNING — could not parse workflow for dynamic fields: {e}")

        required = {
            "workflow":       (workflows, {}),
            "remote_address": ("STRING", {"default": "192.168.1.50:8188", "multiline": False}),
            "prompt":         ("STRING", {"default": "", "multiline": True}),
            "seed":           ("INT",    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "poll_interval":  ("FLOAT",  {"default": 1.0, "min": 0.25, "max": 10.0,  "step": 0.25}),
            "timeout":        ("INT",    {"default": 180, "min": 10,   "max": 600,   "step": 10}),
            "image_index":    ("INT",    {"default": 0,   "min": 0,    "max": 99}),
        }
        required.update(ui_fields)
        return {"required": required, "optional": {"image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "run"
    CATEGORY      = "Karga"
    OUTPUT_NODE   = True

    def run(self, workflow, remote_address, prompt, seed,
            image=None, mask=None, poll_interval=1.0, timeout=180, image_index=0, **ui_values):

        # ── Load workflow ──────────────────────────────────────────────────────
        wf_data = _load_workflow(workflow)
        ui_map  = _find_ui_nodes(wf_data)
        wf      = copy.deepcopy(wf_data)
        print(f"[KargaRemoteWorkflow] Loaded: {workflow}")
        print(f"[KargaRemoteWorkflow] [ui] nodes: {list(ui_map.keys())}")

        # ── Normalise remote address ───────────────────────────────────────────
        remote_address = remote_address.strip().rstrip("/")
        if not remote_address.startswith(("http://", "https://")):
            remote_address = "http://" + remote_address

        # ── Upload input image (optional) ──────────────────────────────────────
        uploaded_filename = _upload_image(remote_address, image) if image is not None else None

        # ── Upload mask (optional) ─────────────────────────────────────────────
        uploaded_mask_filename = _upload_mask(remote_address, mask) if mask is not None else None

        # ── Inject all values into workflow ────────────────────────────────────
        # Build a flat dict: dynamic ui_values + the fixed inputs we own.
        # "seed" maps to the [ui]-tagged RandomNoise node (label="seed").
        injections = dict(ui_values)
        injections["prompt"] = prompt
        injections["seed"]   = seed

        applied   = []
        unmatched = []

        for label, value in injections.items():
            if label not in ui_map:
                unmatched.append(label)
                continue
            node_id, input_key = ui_map[label]
            # Coerce type to match existing value in workflow
            existing = wf[node_id]["inputs"].get(input_key)
            if isinstance(existing, int) and not isinstance(existing, bool):
                try: value = int(value)
                except (ValueError, TypeError): pass
            elif isinstance(existing, float):
                try: value = float(value)
                except (ValueError, TypeError): pass
            wf[node_id]["inputs"][input_key] = value
            applied.append(f"  {label} → node {node_id}.{input_key} = {value!r}")

        # Wire uploaded image filename into the input_image [ui] node
        if uploaded_filename and "input_image" in ui_map:
            node_id, input_key = ui_map["input_image"]
            wf[node_id]["inputs"][input_key] = uploaded_filename
            applied.append(f"  (image) → node {node_id}.{input_key} = {uploaded_filename!r}")

        # Wire uploaded mask filename into the input_mask [ui] node (if provided)
        if uploaded_mask_filename and "input_mask" in ui_map:
            node_id, input_key = ui_map["input_mask"]
            wf[node_id]["inputs"][input_key] = uploaded_mask_filename
            applied.append(f"  (mask) → node {node_id}.{input_key} = {uploaded_mask_filename!r}")

        if applied:
            print("[KargaRemoteWorkflow] Injected:\n" + "\n".join(applied))
        if unmatched:
            print(f"[KargaRemoteWorkflow] WARNING — no [ui] node for: {unmatched}")

        # ── Queue on remote ────────────────────────────────────────────────────
        client_id = str(uuid.uuid4())
        body      = json.dumps({"prompt": wf, "client_id": client_id}).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{remote_address}/prompt",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                result = json.loads(res.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"[KargaRemoteWorkflow] Could not reach {remote_address}: {e}\n"
                "Make sure the remote ComfyUI is running with --listen"
            )

        if "error" in result:
            raise RuntimeError(f"[KargaRemoteWorkflow] Remote rejected prompt: {result['error']}")

        prompt_id = result["prompt_id"]
        queued_at = time.time()
        print(f"[KargaRemoteWorkflow] Queued → prompt_id: {prompt_id}")

        # ── Poll for completion ────────────────────────────────────────────────
        deadline = queued_at + timeout
        print(f"[KargaRemoteWorkflow] Polling for completion...")

        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{remote_address}/history/{prompt_id}", timeout=5
                ) as res:
                    history = json.loads(res.read())
            except urllib.error.URLError as e:
                print(f"[KargaRemoteWorkflow] Poll error (retrying): {e}")
                time.sleep(poll_interval)
                continue

            if prompt_id in history:
                print(f"[KargaRemoteWorkflow] Done in {time.time() - queued_at:.1f}s")
                break

            time.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"[KargaRemoteWorkflow] Job {prompt_id} did not finish within {timeout}s"
            )

        # ── Fetch output frames (images or video) ─────────────────────────────
        all_frames = []
        is_video   = False

        for node_output in history[prompt_id]["outputs"].values():

            # Still images
            if "images" in node_output:
                for img_info in node_output["images"]:
                    params = urllib.parse.urlencode({
                        "filename":  img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type":      img_info.get("type", "output"),
                    })
                    try:
                        with urllib.request.urlopen(
                            f"{remote_address}/view?{params}", timeout=30
                        ) as res:
                            img_data = res.read()
                        img    = Image.open(io.BytesIO(img_data)).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        all_frames.append(img_np)
                    except Exception as e:
                        print(f"[KargaRemoteWorkflow] Failed to fetch image {img_info}: {e}")

            # Video files (VHS mp4 output)
            if "gifs" in node_output:
                for vid_info in node_output["gifs"]:
                    params = urllib.parse.urlencode({
                        "filename":  vid_info["filename"],
                        "subfolder": vid_info.get("subfolder", ""),
                        "type":      vid_info.get("type", "output"),
                    })
                    try:
                        with urllib.request.urlopen(
                            f"{remote_address}/view?{params}", timeout=60
                        ) as res:
                            vid_data = res.read()
                        frames = _frames_from_mp4(vid_data)
                        all_frames.extend(frames)
                        is_video = True
                        print(f"[KargaRemoteWorkflow] Extracted {len(frames)} frames from {vid_info['filename']}")
                    except Exception as e:
                        print(f"[KargaRemoteWorkflow] Failed to fetch video {vid_info}: {e}")

        if not all_frames:
            raise Exception(f"[KargaRemoteWorkflow] No output frames for job {prompt_id}")

        if is_video:
            # Return all frames as a batch tensor [B,H,W,C]
            print(f"[KargaRemoteWorkflow] Returning video batch: {len(all_frames)} frames")
            batch = np.stack(all_frames, axis=0)
            return (torch.from_numpy(batch),)
        else:
            idx = min(image_index, len(all_frames) - 1)
            if idx != image_index:
                print(f"[KargaRemoteWorkflow] image_index {image_index} out of range, using {idx}")
            print(f"[KargaRemoteWorkflow] Returning image {idx + 1}/{len(all_frames)}")
            return (torch.from_numpy(np.expand_dims(all_frames[idx], 0)),)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "KargaRemoteWorkflow": KargaRemoteWorkflow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KargaRemoteWorkflow": "Remote Workflow",
}
