#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-generate fisheye dataset from equirectangular 360° images.

Input structure:
  <src_root>/{train,val}/{blur,sharp}/*.png

Output structure:
  <dst_root>/{train,val}/{blur,sharp}/<stem>_<view>.png

Usage:
  python scripts/generate_fisheye_dataset.py \
      --src /data/360blurry \
      --dst /data/360blurry_fisheye \
      --workers 8
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from basicsr.data.equirect_utils import equirect_to_fisheye_ucm, build_circular_mask

# ── Fixed projection parameters ───────────────────────────────────────────────
OUT_W = 256
OUT_H = 256
XI = 0.9
F_PIX = 220.0
MASK_MODE = "inscribed"

VIEWS = [
    ("front", np.array([0.0,  0.0,  1.0], dtype=np.float32)),
    ("right", np.array([1.0,  0.0,  0.0], dtype=np.float32)),
    ("back",  np.array([0.0,  0.0, -1.0], dtype=np.float32)),
    ("left",  np.array([-1.0, 0.0,  0.0], dtype=np.float32)),
]

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def project_one(src_path: Path, dst_dir: Path) -> list[str]:
    """
    Project a single equirectangular image to 4 fisheye views.
    Returns list of written output paths.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {src_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    written = []
    for view_name, base_dir in VIEWS:
        out = equirect_to_fisheye_ucm(
            img_rgb,
            out_w=OUT_W,
            out_h=OUT_H,
            base_dir=base_dir,
            xi=XI,
            f_pix=F_PIX,
            mask_mode=MASK_MODE,
        )
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        dst_path = dst_dir / f"{src_path.stem}_{view_name}.png"
        cv2.imwrite(str(dst_path), out_bgr)
        written.append(str(dst_path))

    return written


def _worker(args):
    src_path, dst_dir = args
    try:
        project_one(src_path, dst_dir)
        return True, str(src_path)
    except Exception as e:
        return False, f"{src_path}: {e}"


def process_split(src_root: Path, dst_root: Path, split: str, workers: int):
    """Process one split (train or val) across blur and sharp subfolders."""
    tasks = []
    for sub in ("blur", "sharp"):
        src_dir = src_root / split / sub
        dst_dir = dst_root / split / sub
        if not src_dir.exists():
            print(f"  [skip] {src_dir} not found")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for f in files:
            tasks.append((f, dst_dir))

    if not tasks:
        return

    total = len(tasks)
    done = 0
    failed = 0
    print(f"  {split}: {total} images × 4 views = {total * 4} outputs", flush=True)

    with Pool(processes=workers) as pool:
        for ok, info in pool.imap_unordered(_worker, tasks):
            done += 1
            if not ok:
                failed += 1
                print(f"  [ERROR] {info}", flush=True)
            if done % max(1, total // 20) == 0 or done == total:
                pct = done / total * 100
                print(f"  [{split}] {done}/{total} ({pct:.0f}%)  errors={failed}",
                      flush=True)

    if failed:
        print(f"  [WARN] {failed} images failed in split '{split}'")


def main():
    parser = argparse.ArgumentParser(description="Generate fisheye dataset")
    parser.add_argument("--src", default="/data/360blurry",
                        help="Source equirectangular dataset root")
    parser.add_argument("--dst", default="/data/360blurry_fisheye",
                        help="Output fisheye dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to process (default: train val)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"[ERROR] Source root does not exist: {src_root}")
        sys.exit(1)

    print(f"Source : {src_root}")
    print(f"Output : {dst_root}")
    print(f"Params : {OUT_W}×{OUT_H}, xi={XI}, f_pix={F_PIX}, mask={MASK_MODE}")
    print(f"Views  : {[v for v, _ in VIEWS]}")
    print(f"Workers: {args.workers}")
    print()

    for split in args.splits:
        print(f"[{split}]")
        process_split(src_root, dst_root, split, args.workers)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
