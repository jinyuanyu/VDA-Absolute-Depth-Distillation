from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modified_vda import SingleImageVDA
from models.scale_mlp_head import ScaleMLPHead


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_depth_vis(depth: np.ndarray, output_path: Path) -> None:
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    depth_norm = ((depth - d_min) / max(d_max - d_min, 1e-8) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(output_path), vis)


def iter_image_paths(input_path: Path, ext: str) -> Iterable[Path]:
    if input_path.is_dir():
        yield from sorted(input_path.glob(f"*{ext}"))
    else:
        yield input_path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image absolute depth inference with frozen VDA + scale head.")
    parser.add_argument("--config", required=True, help="Path to distill_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained scale-head checkpoint")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output-dir", required=True, help="Directory for absolute-depth outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    preprocess_cfg = cfg["preprocess"]
    scale_cfg = cfg["scale_head"]
    infer_cfg = cfg["inference"]
    mapping_cfg = cfg.get("depth_mapping", {"mode": "linear"})

    device = torch.device(args.device)
    vda_model = SingleImageVDA(
        repo_root=paths["vda_repo_root"],
        checkpoint_path=paths["vda_checkpoint"],
        encoder=model_cfg["encoder"],
        features=int(model_cfg["features"]),
        out_channels=model_cfg["out_channels"],
        input_size=int(preprocess_cfg["input_size"]),
        max_res=int(preprocess_cfg["max_res"]),
        device=device,
    )
    vda_model.freeze_all()
    vda_model.eval()

    scale_head = ScaleMLPHead(
        input_dim=int(scale_cfg["input_dim"]),
        hidden_dims=scale_cfg["hidden_dims"],
        dropout=float(scale_cfg.get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    scale_head.load_state_dict(checkpoint["scale_head_state"], strict=True)
    scale_head.eval()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict[str, Any]] = []
    for image_path in iter_image_paths(input_path, cfg["data"]["image_ext"]):
        image = vda_model.load_rgb_image(image_path)
        result = vda_model.infer_single_image(image)
        feature = result["global_features"].unsqueeze(0).to(device)
        prediction = scale_head(feature)[0]
        scale = float(prediction[0].item())
        shift = float(prediction[1].item())
        relative_depth = result["relative_depth"].detach().cpu().numpy().astype(np.float32)
        mapped_depth = transform_vda_depth(relative_depth, mapping_cfg)
        absolute_depth = scale * mapped_depth + shift
        absolute_depth = np.maximum(absolute_depth, float(infer_cfg["clamp_min_depth"])).astype(np.float32)

        stem = image_path.stem
        np.save(output_dir / f"{stem}.npy", absolute_depth)
        save_depth_vis(absolute_depth, output_dir / f"{stem}.jpg")
        metadata.append(
            {
                "image_path": str(image_path),
                "scale": scale,
                "shift": shift,
                "depth_mapping_mode": str(mapping_cfg.get("mode", "linear")),
                "relative_depth_shape": list(relative_depth.shape),
                "absolute_depth_path": str(output_dir / f"{stem}.npy"),
            }
        )

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_images": len(metadata)}, indent=2))


if __name__ == "__main__":
    main()
