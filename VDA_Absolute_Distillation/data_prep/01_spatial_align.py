from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bilinear_resize(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    resized = cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align VDA depth maps to Depth Pro spatial resolution.")
    parser.add_argument("--config", required=True, help="Path to distill_config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    vda_root = Path(paths["vda_root"])
    depth_pro_root = Path(paths["depth_pro_root"])
    aligned_root = Path(paths["aligned_vda_root"])
    aligned_root.mkdir(parents=True, exist_ok=True)

    allowed_cams = [cam for cam in data_cfg["allowed_cams"] if cam not in set(data_cfg.get("blacklist_cams", []))]
    manifest: Dict[str, Any] = {"aligned_root": str(aligned_root), "cameras": {}}

    for cam in allowed_cams:
        cam_vda_root = vda_root / cam
        cam_depth_root = depth_pro_root / cam
        cam_out_root = aligned_root / cam
        cam_out_root.mkdir(parents=True, exist_ok=True)
        count = 0
        skipped = 0
        for vda_path in sorted(cam_vda_root.glob(f"*{data_cfg['vda_ext']}")):
            frame_id = vda_path.stem
            depth_pro_path = cam_depth_root / f"{frame_id}{data_cfg['depth_pro_ext']}"
            if not depth_pro_path.exists():
                skipped += 1
                continue
            vda_depth = np.load(vda_path).astype(np.float32)
            depth_pro_depth = np.load(depth_pro_path)[data_cfg["depth_pro_npz_key"]].astype(np.float32)
            aligned = bilinear_resize(vda_depth, depth_pro_depth.shape)
            np.save(cam_out_root / f"{frame_id}.npy", aligned)
            count += 1
        manifest["cameras"][cam] = {"aligned_frames": count, "skipped_frames": skipped}

    manifest_path = aligned_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
