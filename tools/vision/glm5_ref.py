#!/usr/bin/env python3
"""Inputs and dump comparison for the GLM-5.3 Flash image tower gate (docs/engine/vision-adapters.md).

  synth    numpy + gguf     small projector files with random weights and the real metadata keys and tensor names
                            (separate and fused q/k/v, with and without biases, pre/post norm, FP16 storage)
  images   numpy + Pillow   the fixed synthetic image set: NAME.png, NAME.rgb (raw RGB8) and cases.txt
  compare  numpy            two row dumps ([rows][width] FP32): per-row cosine and relative L2, optional limits

The rows of this engine come from `strix-glm5-vision encode FILE.gguf NAME.png OUT.rows --f32 NAME.f32`. A second
dump comes from another implementation of the file format run on the same file and the same NAME.rgb pixels.
`synth` writes files of language width 4096 (the engine refuses another width); every other size is small.
"""
import argparse
import os
import sys

import numpy as np


def synth(out):
    import gguf

    def make(name, seed, fused, biases, pre_post, f16=False):
        rng = np.random.default_rng(seed)
        W, H, L, M, P, patch, merge, C, F = 64, 2, 3, 96, 4096, 14, 2, 80, 48
        D = W // H
        w = gguf.GGUFWriter(f"{out}/{name}.gguf", "clip")
        w.add_file_type(1 if f16 else 0)
        w.add_clip_has_vision_encoder(True)
        w.add_clip_projector_type("glm5v")
        w.add_vision_projection_dim(P)
        w.add_vision_image_size(336)
        w.add_vision_patch_size(patch)
        w.add_vision_embedding_length(W)
        w.add_vision_feed_forward_length(M)
        w.add_vision_block_count(L)
        w.add_vision_head_count(H)
        w.add_vision_image_mean([0.48145466, 0.4578275, 0.40821073])
        w.add_vision_image_std([0.26862954, 0.26130258, 0.27577711])
        w.add_vision_use_silu(True)
        w.add_vision_attention_layernorm_eps(1e-5)
        w.add_vision_spatial_merge_size(merge)
        w.add_vision_swiglu_clamp(1.5)
        w.add_vision_min_pixels(16 * (patch * merge) ** 2)
        w.add_vision_max_pixels(144 * (patch * merge) ** 2)

        def t(tensor, shape, scale, offset=0.0):
            a = (offset + scale * rng.standard_normal(shape)).astype(np.float32)
            w.add_tensor(tensor, a.astype(np.float16) if f16 and a.ndim >= 2 else a)

        t("v.patch_embd.weight", (W, 3, patch, patch), 0.05)
        t("v.patch_embd.weight.1", (W, 3, patch, patch), 0.05)
        t("v.patch_embd.bias", (W,), 0.1)
        if pre_post:
            t("v.pre_ln.weight", (W,), 0.2, 1.0)
            t("v.post_ln.weight", (W,), 0.2, 1.0)
        s = 1.0 / np.sqrt(W)
        for layer in range(L):
            p = f"v.blk.{layer}."
            t(p + "ln1.weight", (W,), 0.2, 1.0)
            t(p + "ln2.weight", (W,), 0.2, 1.0)
            if fused:
                t(p + "attn_qkv.weight", (3 * W, W), 2 * s)
                if biases:
                    t(p + "attn_qkv.bias", (3 * W,), 0.1)
            else:
                for part in ("attn_q", "attn_k", "attn_v"):
                    t(p + part + ".weight", (W, W), 2 * s)
                    if biases and part != "attn_k":
                        t(p + part + ".bias", (W,), 0.1)
            t(p + "attn_q_norm.weight", (D,), 0.3, 2.0)
            t(p + "attn_k_norm.weight", (D,), 0.3, 2.0)
            t(p + "attn_out.weight", (W, W), s)
            if biases:
                t(p + "attn_out.bias", (W,), 0.1)
            t(p + "ffn_up.weight", (M, W), 3 * s)
            t(p + "ffn_gate.weight", (M, W), 3 * s)
            t(p + "ffn_down.weight", (W, M), s)
            if biases:
                t(p + "ffn_gate.bias", (M,), 0.1)
                t(p + "ffn_down.bias", (W,), 0.1)
        t("mm.patch_merger.weight", (C, W, merge, merge), 1.0 / np.sqrt(W * 4))
        t("mm.patch_merger.bias", (C,), 0.1)
        t("mm.model.fc.weight", (P, C), 1.0 / np.sqrt(C))
        t("mm.post_norm.weight", (P,), 0.2, 1.0)
        t("mm.post_norm.bias", (P,), 0.1)
        t("mm.up.weight", (F, P), 3.0 / np.sqrt(P))
        t("mm.gate.weight", (F, P), 3.0 / np.sqrt(P))
        t("mm.down.weight", (P, F), 1.0 / np.sqrt(F))
        if biases:
            t("mm.up.bias", (F,), 0.1)
            t("mm.down.bias", (P,), 0.1)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"{out}/{name}.gguf")

    make("sep_bias", 1, False, True, False)
    make("fused_nobias", 2, True, False, False)
    make("fused_bias_prepost", 3, True, True, True)
    make("sep_f16", 4, False, True, True, f16=True)


def images(out):
    from PIL import Image

    rng = np.random.default_rng(7)
    cases = [  # name, width, height, content: what the canvas rule does with it on a 16 to 144 row tower of cell 28
        ("aligned", 112, 84, "noise"),   # cell multiple, under the minimum: scaled up
        ("ragged", 131, 97, "mix"),      # padded right and below
        ("tiny", 23, 17, "grad"),        # far under the minimum
        ("large", 640, 480, "mix"),      # over the maximum: scaled down
        ("wide", 900, 40, "noise"),      # extreme aspect ratios
        ("tall", 30, 700, "grad"),
        ("exactmax", 336, 336, "noise"),  # exactly the maximum, no resample
        ("sliver", 4000, 3, "noise"),    # collapses to one row of cells
    ]
    with open(f"{out}/cases.txt", "w") as listing:
        for name, w, h, kind in cases:
            y, x = np.mgrid[0:h, 0:w]
            if kind == "noise":
                a = rng.integers(0, 256, (h, w, 3))
            elif kind == "grad":
                a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) % 256], -1)
            else:
                a = np.stack([np.sin(x / 7.0) * 127 + 128, np.cos(y / 5.0) * 127 + 128, ((x // 8 + y // 8) % 2) * 255], -1)
            a = a.astype(np.uint8)
            Image.fromarray(a, "RGB").save(f"{out}/{name}.png")
            a.tofile(f"{out}/{name}.rgb")
            listing.write(f"{name} {w} {h}\n")


MAX_DUMP_BYTES = 1 << 30


def compare(args):
    # Width and both file sizes are checked before anything is loaded.
    if args.width <= 0:
        sys.exit("--width must be positive")
    sizes = [os.path.getsize(p) if os.path.isfile(p) else -1 for p in (args.a, args.b)]
    if sizes[0] != sizes[1] or sizes[0] <= 0 or sizes[0] > MAX_DUMP_BYTES or sizes[0] % (4 * args.width):
        sys.exit(f"dumps must be regular files of equal size, whole rows of width {args.width}, at most 1 GiB: {sizes}")
    a = np.fromfile(args.a, np.float32)
    b = np.fromfile(args.b, np.float32)
    if a.size != b.size or a.size == 0 or a.size % args.width:
        sys.exit(f"dump sizes differ or are not rows of width {args.width}: {a.size} and {b.size}")
    a = a.reshape(-1, args.width).astype(np.float64)
    b = b.reshape(-1, args.width).astype(np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        sys.exit("a dump holds a non-finite value")
    cos = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    rel = float(np.sqrt(((a - b) ** 2).sum() / (b * b).sum()))
    print(f"rows {a.shape[0]} cos_min {cos.min():.9f} cos_mean {cos.mean():.9f} rel_l2 {rel:.3e} max_abs {np.abs(a - b).max():.3e}")
    if args.min_cos is not None and cos.min() < args.min_cos:
        sys.exit(f"FAIL: row cosine {cos.min():.9f} under {args.min_cos}")
    if args.max_rel_l2 is not None and rel > args.max_rel_l2:
        sys.exit(f"FAIL: relative L2 {rel:.3e} over {args.max_rel_l2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("synth", "images"):
        sub.add_parser(mode).add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--width", type=int, default=4096)
    c.add_argument("--min-cos", type=float)
    c.add_argument("--max-rel-l2", type=float)
    args = ap.parse_args()
    if args.mode == "synth":
        synth(args.out)
    elif args.mode == "images":
        images(args.out)
    else:
        compare(args)


if __name__ == "__main__":
    main()
