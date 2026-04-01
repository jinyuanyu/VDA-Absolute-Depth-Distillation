from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def list_stems(directory: Path, ext: str) -> List[str]:
    return sorted(p.stem for p in directory.glob(f"*{ext}"))


def load_depth_pro(path: Path, key: str) -> np.ndarray:
    return np.load(path)[key].astype(np.float32)


def resize_to(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def depth_to_color(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip((depth - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    x8 = (x * 255.0).astype(np.uint8)
    return cv2.applyColorMap(x8, cv2.COLORMAP_INFERNO)


def put_label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def sample_values(arr: np.ndarray, stride: int) -> np.ndarray:
    return arr[::stride, ::stride].reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 4-column temporal comparison video.")
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--depthpro-dir", required=True, type=Path)
    parser.add_argument("--vda-dir", required=True, type=Path)
    parser.add_argument("--multi-dir", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--depthpro-key", default="depth")
    parser.add_argument("--rgb-ext", default=".jpg")
    parser.add_argument("--depthpro-ext", default=".npz")
    parser.add_argument("--vda-ext", default=".npy")
    parser.add_argument("--multi-ext", default=".npy")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    stems = list_stems(args.rgb_dir, args.rgb_ext)
    valid_stems: List[str] = []
    for stem in stems:
        if (
            (args.depthpro_dir / f"{stem}{args.depthpro_ext}").exists()
            and (args.vda_dir / f"{stem}{args.vda_ext}").exists()
            and (args.multi_dir / f"{stem}{args.multi_ext}").exists()
        ):
            valid_stems.append(stem)
    if args.max_frames > 0:
        valid_stems = valid_stems[: args.max_frames]
    if not valid_stems:
        raise RuntimeError("No matched frames found across RGB/DepthPro/VDA/Multi directories.")

    # Pass 1: estimate row-shared scale from Depth-Pro and global min/max of VDA inverse.
    dp_samples: List[np.ndarray] = []
    inv_min = float("inf")
    inv_max = float("-inf")
    frame_hw: tuple[int, int] | None = None
    for stem in valid_stems:
        dpro = load_depth_pro(args.depthpro_dir / f"{stem}{args.depthpro_ext}", args.depthpro_key)
        if frame_hw is None:
            frame_hw = dpro.shape
        dp_samples.append(sample_values(dpro, max(args.sample_stride, 1)))

        vda = np.load(args.vda_dir / f"{stem}{args.vda_ext}").astype(np.float32)
        inv = 1.0 / np.maximum(vda, 1e-6)
        inv = resize_to(inv, dpro.shape)
        inv_min = min(inv_min, float(np.min(inv)))
        inv_max = max(inv_max, float(np.max(inv)))

    dp_concat = np.concatenate(dp_samples)
    vmin = float(np.percentile(dp_concat, 2))
    vmax = float(np.percentile(dp_concat, 98))
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    if inv_max - inv_min < 1e-8:
        inv_max = inv_min + 1e-8

    assert frame_hw is not None
    h, w = frame_hw
    out_w = w * 4
    out_h = h
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output_video}")

    prev_multi: np.ndarray | None = None
    temporal_mae: List[float] = []

    for i, stem in enumerate(valid_stems):
        rgb = cv2.imread(str(args.rgb_dir / f"{stem}{args.rgb_ext}"), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        dpro = load_depth_pro(args.depthpro_dir / f"{stem}{args.depthpro_ext}", args.depthpro_key)
        dpro_vis = depth_to_color(dpro, vmin, vmax)

        vda = np.load(args.vda_dir / f"{stem}{args.vda_ext}").astype(np.float32)
        inv = 1.0 / np.maximum(vda, 1e-6)
        inv = resize_to(inv, (h, w))
        inv_255 = (inv - inv_min) / (inv_max - inv_min) * 255.0
        inv_disp = vmin + (inv_255 / 255.0) * (vmax - vmin)
        inv_vis = depth_to_color(inv_disp, vmin, vmax)

        multi = np.load(args.multi_dir / f"{stem}{args.multi_ext}").astype(np.float32)
        multi = resize_to(multi, (h, w))
        multi_vis = depth_to_color(multi, vmin, vmax)

        if prev_multi is not None:
            temporal_mae.append(float(np.mean(np.abs(multi - prev_multi))))
        prev_multi = multi

        put_label(rgb, f"RGB | frame {i+1}/{len(valid_stems)} ({stem})")
        put_label(dpro_vis, f"Depth-Pro | scale [{vmin:.3f}, {vmax:.3f}]")
        put_label(inv_vis, "VDA inverse -> [0,255] -> Depth-Pro scale")
        put_label(multi_vis, "Multi-scene model output")

        canvas = np.concatenate([rgb, dpro_vis, inv_vis, multi_vis], axis=1)
        writer.write(canvas)

    writer.release()

    summary = {
        "num_frames": len(valid_stems),
        "video_path": str(args.output_video),
        "vmin_p2_depthpro": vmin,
        "vmax_p98_depthpro": vmax,
        "vda_inverse_min": inv_min,
        "vda_inverse_max": inv_max,
        "temporal_mae_mean_multi_depth": float(np.mean(temporal_mae)) if temporal_mae else 0.0,
        "temporal_mae_std_multi_depth": float(np.std(temporal_mae)) if temporal_mae else 0.0,
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
