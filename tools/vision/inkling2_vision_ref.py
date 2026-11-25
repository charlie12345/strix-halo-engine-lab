#!/usr/bin/env python3
"""Inkling 2 vision reference: the model's own definition (transformers >= 5.24 Inkling2VisionModel and
Inkling2ImageProcessor) run on the checkpoint's vision tensors (CPU). Needs torch, torchvision, transformers, safetensors,
numpy and pillow. Writes, per image, little-endian float32 dumps for strix-inkling2-vision-gate:

  NAME.pixels.f32    processor output pixel_values [tokens][40][40][3] in [-1, 1]
  NAME.png           the image (synthetic images are written here)
  NAME.grid.json     {"rows": token rows, "cols": token cols, "tokens": N, "width": W, "height": H, "dtype": ...}
  NAME.hidden.f32    encoder output (CLS rows included, CLS first) [1 + 4N][1024]
  NAME.rows.f32      projected image rows (the language model embeddings) [N][2304]

usage: inkling2_vision_ref.py --checkpoint DIR --out DIR [--dtype float32|bfloat16] IMAGE...  (IMAGE may be synth:WxH[:seed])
"""
import argparse, json, os, re, sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel


def rename(key):
    # conversion_mapping.py "inkling2" rules, vision part only
    k = key.replace("model.visual.", "")
    k = k.replace("backbone.embeddings.position_embedding", "embeddings.position_embedding.weight")
    k = k.replace("backbone.embeddings.", "embeddings.")
    k = k.replace("backbone.ln_pre.", "pre_layernorm.")
    k = k.replace("backbone.layers.", "layers.")
    k = k.replace("proj_in.", "merger.linear_1.").replace("proj_out.", "merger.linear_2.")
    for a, b in ((".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."), (".mlp.c_fc.", ".mlp.fc1."),
                 (".mlp.c_proj.", ".mlp.fc2."), (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2")):
        k = k.replace(a, b)
    return k


def load_model(checkpoint, dtype):
    cfg_path = os.path.join(checkpoint, "config.json")
    if os.path.getsize(cfg_path) > MAX_CONFIG_BYTES:
        sys.exit("config.json is larger than 1 MiB")
    with open(cfg_path, encoding="utf-8") as f:
        cfg_all = json.load(f)
    vc = dict(cfg_all["vision_config"])
    vc.pop("model_type", None)
    config = Inkling2VisionConfig(**vc)
    model = Inkling2VisionModel(config).eval()
    state = {}
    path = os.path.join(checkpoint, "model-00001-of-00001.safetensors")
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if key.startswith("model.visual."):
                state[rename(key)] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m]
    if missing or unexpected:
        sys.exit(f"weight mismatch: missing {missing} unexpected {unexpected}")
    return model.to(dtype), config


MAX_CONFIG_BYTES = 1 << 20
MAX_IMAGE_BYTES = 20 << 20      # the gate's decoder limit
MAX_IMAGE_PIXELS = 40_000_000
MAX_SYNTH_SIDE = 4096
SYNTH = re.compile(r"synth:([0-9]{1,5})x([0-9]{1,5})(?::([0-9]{1,9}))?")
SIMPLE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")


def synth_fields(spec):
    """synth:WxH[:seed], 1 <= W, H <= 4096; anything else is refused."""
    m = SYNTH.fullmatch(spec)
    if not m:
        sys.exit(f"{spec!r}: expected synth:WxH[:seed]")
    w, h, seed = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if not (1 <= w <= MAX_SYNTH_SIDE and 1 <= h <= MAX_SYNTH_SIDE):
        sys.exit(f"{spec!r}: each side must be 1..{MAX_SYNTH_SIDE}")
    return w, h, seed


def synth(w, h, seed):
    rng = np.random.default_rng(seed)
    # smooth gradients plus noise and a few flat blocks: exercises padding, edges and flat areas
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 / max(w - 1, 1)), (y * 255 / max(h - 1, 1)), ((x + y) % 64) * 4], -1)
    img = img + rng.normal(0, 12, img.shape)
    img[h // 4: h // 2, w // 4: w // 2] = [200, 30, 30]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    torch.set_num_threads(os.cpu_count() or 8)
    model, config = load_model(a.checkpoint, dtype)
    proc = Inkling2ImageProcessor()
    for spec in a.images:
        if spec.startswith("synth:"):
            w, h, seed = synth_fields(spec)
            image, name, source = synth(w, h, seed), f"synth_{w}x{h}_{seed}", f"synth:{w}x{h}:{seed}"
            image.save(os.path.join(a.out, name + ".png"))
        else:
            name, source = os.path.splitext(os.path.basename(spec))[0], os.path.basename(spec)
            if not SIMPLE_NAME.fullmatch(name):
                sys.exit(f"{spec}: the file name must be a simple name ([A-Za-z0-9._-])")
            if not os.path.isfile(spec) or os.path.getsize(spec) > MAX_IMAGE_BYTES:
                sys.exit(f"{spec}: not a regular file of at most {MAX_IMAGE_BYTES} bytes")
            image = Image.open(spec)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                sys.exit(f"{spec}: more than {MAX_IMAGE_PIXELS} pixels")
            if image.format != "PNG":
                sys.exit(f"{spec}: the gate decodes PNG only (lossless, same pixels as PIL)")
            import shutil
            shutil.copyfile(spec, os.path.join(a.out, name + ".png"))
        feats = proc(images=[image], return_tensors="pt")
        pixels, grid = feats["pixel_values"], feats["image_grid_thw"]
        with torch.no_grad():
            out = model(pixel_values=pixels.to(dtype), image_grid_thw=grid)
        rows = out.pooler_output.float().numpy()
        hidden = out.last_hidden_state.float().numpy()
        t, gr, gc = (int(v) for v in grid[0])
        base = os.path.join(a.out, name)
        pixels.float().numpy().astype("<f4").tofile(base + ".pixels.f32")
        hidden.astype("<f4").tofile(base + ".hidden.f32")
        rows.astype("<f4").tofile(base + ".rows.f32")
        json.dump({"rows": gr, "cols": gc, "tokens": gr * gc, "width": image.width, "height": image.height,
                   "dtype": a.dtype, "source": source}, open(base + ".grid.json", "w"))
        print(f"{name}: {image.width}x{image.height} -> {gr}x{gc} = {gr * gc} tokens, rows {rows.shape}, "
              f"|rows| mean {np.abs(rows).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
