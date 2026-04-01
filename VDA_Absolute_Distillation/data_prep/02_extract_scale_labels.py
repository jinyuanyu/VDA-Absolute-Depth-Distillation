from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bilinear_resize(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def transform_vda_depth(vda_depth: np.ndarray, mapping_cfg: Dict[str, Any]) -> np.ndarray:
    mode = str(mapping_cfg.get("mode", "linear")).lower()
    if mode == "linear":
        return vda_depth.astype(np.float32)
    if mode == "reciprocal_linear":
        eps = float(mapping_cfg.get("reciprocal_eps", 1.0e-6))
        if eps <= 0:
            raise ValueError("depth_mapping.reciprocal_eps must be > 0")
        return (1.0 / np.maximum(vda_depth.astype(np.float32), eps)).astype(np.float32)
    raise ValueError(f"Unsupported depth mapping mode: {mode}")


def solve_scale_shift(vda_depth: np.ndarray, depth_pro_depth: np.ndarray) -> tuple[float, float, float, int]:
    mask = np.isfinite(vda_depth) & np.isfinite(depth_pro_depth)
    v = vda_depth[mask].reshape(-1).astype(np.float64)
    d = depth_pro_depth[mask].reshape(-1).astype(np.float64)
    if v.size == 0:
        raise ValueError("No valid pixels found for least-squares fitting.")

    # Closed-form least squares for y ~= s*x + t to avoid allocating a huge
    # design matrix per frame.
    a00 = float(np.sum(v * v))
    a01 = float(np.sum(v))
    a11 = float(v.size)
    b0 = float(np.sum(v * d))
    b1 = float(np.sum(d))

    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-12:
        scale = 1.0
        shift = float(np.mean(d - v))
    else:
        scale = (a11 * b0 - a01 * b1) / det
        shift = (-a01 * b0 + a00 * b1) / det

    residual = float(np.mean((scale * v + shift - d) ** 2))
    return scale, shift, residual, int(v.size)


def maybe_plot(camera: str, rows: List[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [row["frame_id"] for row in rows]
    scales = [row["scale"] for row in rows]
    shifts = [row["shift"] for row in rows]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(frames, scales)
    plt.title(f"{camera} scale")
    plt.xticks(rotation=90)
    plt.subplot(1, 2, 2)
    plt.plot(frames, shifts)
    plt.title(f"{camera} shift")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / f"{camera}_scale_shift.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract offline scale and shift labels from VDA and Depth Pro.")
    parser.add_argument("--config", required=True, help="Path to distill_config.yaml")
    parser.add_argument("--plot", action="store_true", help="Save scale/shift plots per camera")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    mapping_cfg = cfg.get("depth_mapping", {"mode": "linear"})

    images_root = Path(paths["images_root"])
    vda_root = Path(paths["vda_root"])
    depth_pro_root = Path(paths["depth_pro_root"])
    labels_json = Path(paths["labels_json"])
    labels_csv = Path(paths["labels_csv"])
    plots_root = Path(paths["plots_root"])

    labels_json.parent.mkdir(parents=True, exist_ok=True)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    allowed_cams = [cam for cam in data_cfg["allowed_cams"] if cam not in set(data_cfg.get("blacklist_cams", []))]
    samples: List[Dict[str, Any]] = []
    per_camera_rows: Dict[str, List[Dict[str, Any]]] = {}

    for cam in allowed_cams:
        cam_rows: List[Dict[str, Any]] = []
        for depth_pro_path in sorted((depth_pro_root / cam).glob(f"*{data_cfg['depth_pro_ext']}")):
            frame_id = depth_pro_path.stem
            vda_path = vda_root / cam / f"{frame_id}{data_cfg['vda_ext']}"
            image_path = images_root / cam / f"{frame_id}{data_cfg['image_ext']}"
            if not vda_path.exists() or not image_path.exists():
                continue

            vda_depth = np.load(vda_path).astype(np.float32)
            depth_pro_depth = np.load(depth_pro_path)[data_cfg["depth_pro_npz_key"]].astype(np.float32)
            vda_up = bilinear_resize(vda_depth, depth_pro_depth.shape)
            vda_mapped = transform_vda_depth(vda_up, mapping_cfg)
            scale, shift, residual_mse, valid_pixels = solve_scale_shift(vda_mapped, depth_pro_depth)
            row = {
                "camera": cam,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "vda_path": str(vda_path),
                "depth_pro_path": str(depth_pro_path),
                "vda_shape": list(vda_depth.shape),
                "depth_pro_shape": list(depth_pro_depth.shape),
                "depth_mapping_mode": str(mapping_cfg.get("mode", "linear")),
                "scale": scale,
                "shift": shift,
                "residual_mse": residual_mse,
                "valid_pixels": valid_pixels,
            }
            cam_rows.append(row)
            samples.append(row)
        per_camera_rows[cam] = cam_rows
        if args.plot and cam_rows:
            maybe_plot(cam, cam_rows, plots_root)

    payload = {
        "meta": {
            "allowed_cams": allowed_cams,
            "blacklist_cams": data_cfg.get("blacklist_cams", []),
            "depth_pro_npz_key": data_cfg["depth_pro_npz_key"],
            "depth_mapping": {
                "mode": str(mapping_cfg.get("mode", "linear")),
                "reciprocal_eps": float(mapping_cfg.get("reciprocal_eps", 1.0e-6)),
            },
            "sample_count": len(samples),
        },
        "samples": samples,
    }
    labels_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(labels_csv, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "camera",
            "frame_id",
            "image_path",
            "vda_path",
            "depth_pro_path",
            "depth_mapping_mode",
            "scale",
            "shift",
            "residual_mse",
            "valid_pixels",
            "vda_shape",
            "depth_pro_shape",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

    summary = {
        "sample_count": len(samples),
        "cameras": {camera: len(rows) for camera, rows in per_camera_rows.items()},
        "labels_json": str(labels_json),
        "labels_csv": str(labels_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
