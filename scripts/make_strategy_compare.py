from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/media/a1/16THDD/YJY/VDA_Absolute_Distillation")
OUT_DIR = ROOT / "artifacts" / "strategy_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = [
    {
        "name": "Scene-1 coffee_martini/cam19/0000",
        "rgb": Path("/media/a1/16THDD/XZB/DyNeRF/coffee_martini/images/cam19/0000.jpg"),
        "depth_pro": Path("/media/a1/16THDD/XZB/DyNeRF/coffee_martini/raw_depth_pro_depth/cam19/0000.npz"),
        "vda_raw": Path("/media/a1/16THDD/XZB/DyNeRF/coffee_martini/raw_vda_depth/cam19/0000.npy"),
        "single_abs": ROOT / "artifacts/strategy_compare/single_coffee/0000.npy",
        "multi_abs": ROOT / "artifacts/strategy_compare/multi_coffee/0000.npy",
    },
    {
        "name": "Scene-2 actor2_3/00/000000",
        "rgb": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/images/00/000000.jpg"),
        "depth_pro": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/raw_depth_pro_depth/00/000000.npz"),
        "vda_raw": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/raw_vda_depth/00/000000.npy"),
        "single_abs": ROOT / "artifacts/strategy_compare/single_actor/000000.npy",
        "multi_abs": ROOT / "artifacts/strategy_compare/multi_actor/000000.npy",
    },
]


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def depth_vis(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    finite = np.isfinite(x)
    if np.any(finite):
        p2, p98 = np.percentile(x[finite], [2, 98])
    else:
        p2, p98 = 0.0, 1.0
    if p98 - p2 < 1e-8:
        p2, p98 = float(np.min(x)), float(np.max(x) + 1e-8)
    x = np.clip((x - p2) / (p98 - p2), 0.0, 1.0)
    return plt.cm.inferno(x)[..., :3]


def main() -> None:
    metrics = []
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    col_titles = ["RGB", "Depth-Pro", "VDA Inverse", "Single-Scene Model", "Multi-Scene Model"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    for i, scene in enumerate(SCENES):
        rgb_bgr = cv2.imread(str(scene["rgb"]), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Missing RGB image: {scene['rgb']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        dpro = np.load(scene["depth_pro"])["depth"].astype(np.float32)
        hdp, wdp = dpro.shape

        vda = np.load(scene["vda_raw"]).astype(np.float32)
        vda_inv = 1.0 / np.maximum(vda, 1e-6)
        vda_inv = resize_to(vda_inv, (hdp, wdp))

        d_single = np.load(scene["single_abs"]).astype(np.float32)
        d_multi = np.load(scene["multi_abs"]).astype(np.float32)
        d_single = resize_to(d_single, (hdp, wdp))
        d_multi = resize_to(d_multi, (hdp, wdp))

        mse_single = float(np.mean((d_single - dpro) ** 2))
        mse_multi = float(np.mean((d_multi - dpro) ** 2))
        improvement = float((mse_single - mse_multi) / max(mse_single, 1e-8) * 100.0)

        metrics.append(
            {
                "scene": scene["name"],
                "mse_single_vs_depth_pro": mse_single,
                "mse_multi_vs_depth_pro": mse_multi,
                "improvement(%)": improvement,
            }
        )

        rgb_show = resize_to(rgb.astype(np.float32), (hdp, wdp)).astype(np.uint8)

        axes[i, 0].imshow(rgb_show)
        axes[i, 1].imshow(depth_vis(dpro))
        axes[i, 2].imshow(depth_vis(vda_inv))
        axes[i, 3].imshow(depth_vis(d_single))
        axes[i, 4].imshow(depth_vis(d_multi))

        axes[i, 3].set_xlabel(f"MSE={mse_single:.4f}", fontsize=11)
        axes[i, 4].set_xlabel(f"MSE={mse_multi:.4f}", fontsize=11)

        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 0].set_ylabel(scene["name"], fontsize=12)

    fig.suptitle("Old vs New Training Strategy (Latest Weights)", fontsize=18)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_png = OUT_DIR / "compare_rgb_depthpro_vdainv_single_vs_multi_2scene_en.png"
    out_json = OUT_DIR / "compare_rgb_depthpro_vdainv_single_vs_multi_2scene_metrics.json"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "figure": str(out_png),
                "metrics": str(out_json),
                "summary": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
