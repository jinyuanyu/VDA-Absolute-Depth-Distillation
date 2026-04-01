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
        "multi_abs": ROOT / "artifacts/strategy_compare/multi_coffee/0000.npy",
    },
    {
        "name": "Scene-2 actor2_3/00/000000",
        "rgb": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/images/00/000000.jpg"),
        "depth_pro": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/raw_depth_pro_depth/00/000000.npz"),
        "vda_raw": Path("/media/a1/16THDD/XZB/enerf_outdoor/actor2_3/raw_vda_depth/00/000000.npy"),
        "multi_abs": ROOT / "artifacts/strategy_compare/multi_actor/000000.npy",
    },
]


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def main() -> None:
    metrics = []
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    col_titles = ["RGB", "Depth-Pro (shared scale)", "VDA Inverse [0-255] (shared scale)", "Multi-Scene (shared scale)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13)

    for i, scene in enumerate(SCENES):
        rgb_bgr = cv2.imread(str(scene["rgb"]), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Missing RGB image: {scene['rgb']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        dpro = np.load(scene["depth_pro"])["depth"].astype(np.float32)
        hdp, wdp = dpro.shape
        dpro_finite = dpro[np.isfinite(dpro)]
        vmin = float(np.percentile(dpro_finite, 2))
        vmax = float(np.percentile(dpro_finite, 98))
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8

        vda = np.load(scene["vda_raw"]).astype(np.float32)
        vda_inv = 1.0 / np.maximum(vda, 1e-6)
        vda_inv = resize_to(vda_inv, (hdp, wdp))
        inv_min = float(np.min(vda_inv))
        inv_max = float(np.max(vda_inv))
        if inv_max - inv_min < 1e-8:
            vda_inv_255 = np.zeros_like(vda_inv, dtype=np.float32)
        else:
            vda_inv_255 = ((vda_inv - inv_min) / (inv_max - inv_min) * 255.0).astype(np.float32)
        # Keep row colorbar on Depth-Pro range by remapping [0,255] to [vmin,vmax] for display.
        vda_inv_disp = vmin + (vda_inv_255 / 255.0) * (vmax - vmin)

        d_multi = np.load(scene["multi_abs"]).astype(np.float32)
        d_multi = resize_to(d_multi, (hdp, wdp))

        mse_multi = float(np.mean((d_multi - dpro) ** 2))
        mae_multi = float(np.mean(np.abs(d_multi - dpro)))
        metrics.append(
            {
                "scene": scene["name"],
                "depth_pro_vmin_p2": vmin,
                "depth_pro_vmax_p98": vmax,
                "mse_multi_vs_depth_pro": mse_multi,
                "mae_multi_vs_depth_pro": mae_multi,
                "depth_pro_mean": float(np.mean(dpro)),
                "multi_mean": float(np.mean(d_multi)),
                "vda_inverse_255_mean": float(np.mean(vda_inv_255)),
            }
        )

        rgb_show = resize_to(rgb.astype(np.float32), (hdp, wdp)).astype(np.uint8)
        axes[i, 0].imshow(rgb_show)
        im1 = axes[i, 1].imshow(dpro, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(vda_inv_disp, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[i, 3].imshow(d_multi, cmap="inferno", vmin=vmin, vmax=vmax)

        axes[i, 3].set_xlabel(f"MSE={mse_multi:.4f} | MAE={mae_multi:.4f}", fontsize=10)
        axes[i, 0].set_ylabel(scene["name"], fontsize=11)

    # Reserve right margin and place colorbars at the far right to avoid overlap.
    fig.subplots_adjust(left=0.04, right=0.88, top=0.92, bottom=0.06, wspace=0.04, hspace=0.08)
    for i in range(2):
        pos = axes[i, 3].get_position()
        cax = fig.add_axes([0.90, pos.y0, 0.012, pos.height])
        cbar = fig.colorbar(axes[i, 1].images[0], cax=cax)
        cbar.set_label("Depth scale")

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle("4-Column Comparison with Row-wise Shared Color Scale", fontsize=16)

    out_png = OUT_DIR / "compare_rgb_depthpro_vdainv_multi_2scene_4cols_shared_scale.png"
    out_json = OUT_DIR / "compare_rgb_depthpro_vdainv_multi_2scene_4cols_shared_scale_metrics.json"
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
