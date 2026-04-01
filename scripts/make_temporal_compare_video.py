from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 4-column temporal comparison video.")
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--depth-pro-dir", required=True)
    p.add_argument("--vda-dir", required=True)
    p.add_argument("--pred-dir", required=True, help="Predicted absolute depth directory (*.npy from inference_abs_vda.py)")
    p.add_argument("--output-video", required=True)
    p.add_argument("--output-metrics", required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--depth-key", default="depth")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--depth-pro-ext", default=".npz")
    p.add_argument("--vda-ext", default=".npy")
    p.add_argument("--pred-ext", default=".npy")
    p.add_argument("--sample-per-frame", type=int, default=2048)
    p.add_argument("--max-frames", type=int, default=-1)
    return p.parse_args()


def sort_key(frame_id: str):
    return (0, int(frame_id)) if frame_id.isdigit() else (1, frame_id)


def colorize_depth_bgr(depth: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    x = depth.astype(np.float32)
    if dmax - dmin < 1e-8:
        dmax = dmin + 1e-8
    x = np.clip((x - dmin) / (dmax - dmin), 0.0, 1.0)
    x8 = (x * 255.0).astype(np.uint8)
    return cv2.applyColorMap(x8, cv2.COLORMAP_INFERNO)


def put_panel_title(panel_bgr: np.ndarray, title: str) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    bar_h = max(30, int(0.06 * h))
    cv2.rectangle(out, (0, 0), (w, bar_h), (20, 20, 20), thickness=-1)
    cv2.putText(out, title, (10, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
    return out


def iter_frame_ids(pred_dir: Path, pred_ext: str) -> list[str]:
    ids = [p.stem for p in pred_dir.glob(f"*{pred_ext}")]
    return sorted(ids, key=sort_key)


def sample_values(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    flat = arr.reshape(-1)
    if flat.size <= n:
        return flat.astype(np.float32)
    idx = rng.integers(0, flat.size, size=n)
    return flat[idx].astype(np.float32)


def build_color_scales(
    frame_ids: Iterable[str],
    depth_pro_dir: Path,
    vda_dir: Path,
    depth_key: str,
    depth_pro_ext: str,
    vda_ext: str,
    sample_per_frame: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(42)
    depth_samples: list[np.ndarray] = []
    inv_samples: list[np.ndarray] = []

    for fid in frame_ids:
        dpro = np.load(depth_pro_dir / f"{fid}{depth_pro_ext}")[depth_key].astype(np.float32)
        vda = np.load(vda_dir / f"{fid}{vda_ext}").astype(np.float32)
        inv = 1.0 / np.maximum(vda, 1.0e-6)

        depth_samples.append(sample_values(dpro, sample_per_frame, rng))
        inv_samples.append(sample_values(inv, sample_per_frame, rng))

    depth_concat = np.concatenate(depth_samples, axis=0)
    inv_concat = np.concatenate(inv_samples, axis=0)

    dmin = float(np.percentile(depth_concat, 2))
    dmax = float(np.percentile(depth_concat, 98))
    inv_lo = float(np.percentile(inv_concat, 1))
    inv_hi = float(np.percentile(inv_concat, 99))
    if inv_hi - inv_lo < 1e-8:
        inv_hi = inv_lo + 1e-8
    return dmin, dmax, inv_lo, inv_hi


def main() -> None:
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    depth_pro_dir = Path(args.depth_pro_dir)
    vda_dir = Path(args.vda_dir)
    pred_dir = Path(args.pred_dir)
    out_video = Path(args.output_video)
    out_metrics = Path(args.output_metrics)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = iter_frame_ids(pred_dir, args.pred_ext)
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError(f"No prediction frames found in {pred_dir}")

    valid_ids: list[str] = []
    for fid in frame_ids:
        if (rgb_dir / f"{fid}{args.image_ext}").exists() and (depth_pro_dir / f"{fid}{args.depth_pro_ext}").exists() and (
            vda_dir / f"{fid}{args.vda_ext}"
        ).exists():
            valid_ids.append(fid)
    frame_ids = valid_ids
    if not frame_ids:
        raise RuntimeError("No valid frame intersection among RGB/Depth-Pro/VDA/Prediction directories.")

    dmin, dmax, inv_lo, inv_hi = build_color_scales(
        frame_ids,
        depth_pro_dir,
        vda_dir,
        args.depth_key,
        args.depth_pro_ext,
        args.vda_ext,
        args.sample_per_frame,
    )

    # Use first frame to determine output size
    fid0 = frame_ids[0]
    d0 = np.load(depth_pro_dir / f"{fid0}{args.depth_pro_ext}")[args.depth_key].astype(np.float32)
    h, w = d0.shape
    canvas_w = w * 4
    canvas_h = h

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_video}")

    frame_metrics: list[dict] = []
    prev_pred: np.ndarray | None = None

    for idx, fid in enumerate(frame_ids):
        rgb = cv2.imread(str(rgb_dir / f"{fid}{args.image_ext}"), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        dpro = np.load(depth_pro_dir / f"{fid}{args.depth_pro_ext}")[args.depth_key].astype(np.float32)
        vda = np.load(vda_dir / f"{fid}{args.vda_ext}").astype(np.float32)
        pred = np.load(pred_dir / f"{fid}{args.pred_ext}").astype(np.float32)

        if vda.shape != dpro.shape:
            vda = cv2.resize(vda, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if pred.shape != dpro.shape:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # VDA inverse -> [0,255] (global sequence scaling), then remap to Depth-Pro display range.
        inv = 1.0 / np.maximum(vda, 1.0e-6)
        inv255 = np.clip((inv - inv_lo) / (inv_hi - inv_lo), 0.0, 1.0) * 255.0
        inv_disp = dmin + (inv255 / 255.0) * (dmax - dmin)

        dpro_bgr = colorize_depth_bgr(dpro, dmin, dmax)
        inv_bgr = colorize_depth_bgr(inv_disp, dmin, dmax)
        pred_bgr = colorize_depth_bgr(pred, dmin, dmax)

        p0 = put_panel_title(rgb, "RGB")
        p1 = put_panel_title(dpro_bgr, f"Depth-Pro [{dmin:.2f}, {dmax:.2f}]")
        p2 = put_panel_title(inv_bgr, "VDA Inverse [0-255]")
        p3 = put_panel_title(pred_bgr, "Multi-Scene Result")

        mse = float(np.mean((pred - dpro) ** 2))
        mae = float(np.mean(np.abs(pred - dpro)))
        jitter = float(np.mean(np.abs(pred - prev_pred))) if prev_pred is not None else 0.0
        prev_pred = pred.copy()
        cv2.putText(
            p3,
            f"frame={fid}  MSE={mse:.4f}  MAE={mae:.4f}  jitter={jitter:.4f}",
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        frame_metrics.append(
            {
                "index": idx,
                "frame_id": fid,
                "mse_vs_depth_pro": mse,
                "mae_vs_depth_pro": mae,
                "jitter_l1_vs_prev": jitter,
            }
        )

        canvas = np.concatenate([p0, p1, p2, p3], axis=1)
        writer.write(canvas)

    writer.release()

    mse_vals = np.array([m["mse_vs_depth_pro"] for m in frame_metrics], dtype=np.float64)
    mae_vals = np.array([m["mae_vs_depth_pro"] for m in frame_metrics], dtype=np.float64)
    jit_vals = np.array([m["jitter_l1_vs_prev"] for m in frame_metrics[1:]], dtype=np.float64) if len(frame_metrics) > 1 else np.array([0.0])
    summary = {
        "num_frames": len(frame_metrics),
        "output_video": str(out_video),
        "depth_pro_display_range": [dmin, dmax],
        "vda_inverse_global_range": [inv_lo, inv_hi],
        "mse_mean": float(np.mean(mse_vals)),
        "mse_p95": float(np.percentile(mse_vals, 95)),
        "mae_mean": float(np.mean(mae_vals)),
        "jitter_mean": float(np.mean(jit_vals)),
        "jitter_p95": float(np.percentile(jit_vals, 95)),
    }
    payload = {"summary": summary, "frames": frame_metrics}
    out_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
