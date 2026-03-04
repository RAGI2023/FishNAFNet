#!/usr/bin/env python3
"""
Generate motion blur dataset from panoramic videos.

Output structure:
  {output}/{split}/sharp/000000.png  <- GT (center frame)
  {output}/{split}/blur/000000.png   <- mean of 2t+1 frames

Blur synthesis: simple mean of (GT-t ... GT ... GT+t), matching GoPro convention.
GT frames are selected non-overlappingly with stride 2t+1.
"""

import cv2
import numpy as np
import argparse
import random
import multiprocessing as mp
from pathlib import Path
from collections import deque
from tqdm import tqdm


def get_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def count_gt_pairs(n_frames, t):
    stride = 2 * t + 1
    return len(range(t, n_frames - t, stride))


def process_video(args):
    video_path, split, output_root, t, file_offset = args

    sharp_dir = Path(output_root) / split / "sharp"
    blur_dir = Path(output_root) / split / "blur"

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stride = 2 * t + 1
    gt_indices = set(range(t, n_frames - t, stride))
    gt_order = {idx: i for i, idx in enumerate(sorted(gt_indices))}

    if not gt_indices:
        cap.release()
        return split, 0

    buffer = deque(maxlen=stride)
    count = 0

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)

        if len(buffer) == stride:
            center_idx = frame_idx - t
            if center_idx in gt_indices:
                sharp_frame = buffer[t]
                blur_frame = np.clip(
                    np.mean([f.astype(np.float32) for f in buffer], axis=0),
                    0, 255
                ).astype(np.uint8)

                fname = f"{file_offset + gt_order[center_idx]:06d}.png"
                cv2.imwrite(str(sharp_dir / fname), sharp_frame)
                cv2.imwrite(str(blur_dir / fname), blur_frame)
                count += 1

    cap.release()
    return split, count


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion blur dataset from panoramic videos"
    )
    parser.add_argument(
        "--input", default="/data/360x_dataset_LR/panoramic",
        help="Input directory containing mp4 files"
    )
    parser.add_argument(
        "--output", default="/data/360blurry",
        help="Output dataset root directory"
    )
    parser.add_argument(
        "--t", type=int, default=8,
        help="Frames before/after GT to average (blur window = 2t+1, default: 8)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"No mp4 files found in {input_dir}")
        return

    print(f"Found {len(videos)} videos")

    # 8:1:1 split by video
    rng = random.Random(args.seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    n_train = n - n_val - n_test

    split_map = {}
    for v in shuffled[:n_train]:
        split_map[v] = "train"
    for v in shuffled[n_train:n_train + n_val]:
        split_map[v] = "val"
    for v in shuffled[n_train + n_val:]:
        split_map[v] = "test"

    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    print(f"Blur window: {2 * args.t + 1} frames (t={args.t})\n")

    # Scan frame counts to compute per-video filename offsets
    print("Scanning video metadata...")
    with mp.Pool(args.workers) as pool:
        frame_counts = list(tqdm(
            pool.imap(get_frame_count, [str(v) for v in shuffled]),
            total=len(shuffled), desc="Scanning"
        ))

    # Compute offsets within each split
    split_offsets = {"train": {}, "val": {}, "test": {}}
    split_cursors = {"train": 0, "val": 0, "test": 0}
    for v, n_frames in zip(shuffled, frame_counts):
        split = split_map[v]
        split_offsets[split][v] = split_cursors[split]
        split_cursors[split] += count_gt_pairs(n_frames, args.t)

    # Create output directories
    for split in ["train", "val", "test"]:
        (Path(args.output) / split / "sharp").mkdir(parents=True, exist_ok=True)
        (Path(args.output) / split / "blur").mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(v), split_map[v], args.output, args.t, split_offsets[split_map[v]][v])
        for v in shuffled
    ]

    print("Processing videos...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_video, tasks),
            total=len(tasks), desc="Processing"
        ))

    split_counts = {"train": 0, "val": 0, "test": 0}
    for split, count in results:
        split_counts[split] += count

    total = sum(split_counts.values())
    print(f"\nDone! Generated {total} sharp/blur pairs")
    for split in ["train", "val", "test"]:
        print(f"  {split:5s}: {split_counts[split]} pairs")


if __name__ == "__main__":
    main()
