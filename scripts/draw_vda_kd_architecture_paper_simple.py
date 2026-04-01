from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc, ec="#4d4d4d", lw=1.2, fs=10, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def add_arrow(ax, x1, y1, x2, y2, text=None, dashed=False, color="#4d4d4d"):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.3,
        linestyle="--" if dashed else "-",
        color=color,
    )
    ax.add_patch(arr)
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.15,
            text,
            fontsize=8.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"),
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 7.6))
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 10.0)
    ax.axis("off")

    # Colors
    c_input = "#DCEEFF"
    c_student = "#CFE8FF"
    c_teacher = "#FFE7C2"
    c_train = "#D9F5E5"
    c_infer = "#EAF8F0"

    # Background bands (clean train/inference split)
    add_box(ax, 0.4, 7.4, 16.1, 1.9, "", "#F8F4EA", ec="#F8F4EA", lw=0.0)
    add_box(ax, 0.4, 3.8, 16.1, 2.9, "", "#F0F8FF", ec="#F0F8FF", lw=0.0)
    ax.text(0.7, 9.45, "Training supervision path", fontsize=10, color="#8E5B1A", weight="bold")
    ax.text(0.7, 6.95, "Inference path", fontsize=10, color="#2B5E85", weight="bold")

    # Title
    ax.text(0.55, 9.78, "VDA Knowledge Distillation for Absolute Depth", fontsize=18, weight="bold")

    # Core blocks
    add_box(ax, 0.9, 5.0, 2.7, 1.2, "Input RGB", c_input, fs=10, weight="bold")
    add_box(ax, 4.0, 4.8, 2.9, 1.6, "Frozen VDA\nEncoder + Decoder", c_student, fs=9)

    add_box(ax, 7.4, 5.45, 2.9, 0.95, "Global feature f", c_student, fs=9)
    add_box(ax, 7.4, 4.05, 2.9, 0.95, "Relative depth D_vda", c_student, fs=9)

    add_box(ax, 11.2, 5.35, 2.9, 1.05, "Scale Head\n(s_hat, t_hat)", c_train, fs=9, weight="bold")
    add_box(ax, 11.2, 4.00, 2.9, 1.05, "Fusion\nD_abs = s_hat·φ(D_vda)+t_hat", c_infer, fs=8.6)
    add_box(ax, 14.6, 4.00, 1.9, 1.05, "Output\nD_abs", c_infer, fs=9, weight="bold")

    # Teacher pseudo-label branch
    add_box(ax, 4.0, 8.0, 2.9, 1.0, "Depth-Pro\nD_dp", c_teacher, fs=9, weight="bold")
    add_box(ax, 7.4, 7.8, 2.9, 1.35, "Align + Mapping\nresize + φ(D_vda)", c_teacher, fs=8.6)
    add_box(ax, 11.2, 8.0, 2.9, 1.0, "Pseudo labels\n(s*, t*)", c_teacher, fs=9)
    add_box(ax, 14.6, 8.0, 1.9, 1.0, "Loss\nMSE", c_train, fs=9, weight="bold")

    # Main clean arrows (mostly straight, no crossing)
    add_arrow(ax, 3.6, 5.6, 4.0, 5.6)
    add_arrow(ax, 6.9, 5.9, 7.4, 5.9, text="feature")
    add_arrow(ax, 6.9, 4.5, 7.4, 4.5, text="depth")
    add_arrow(ax, 10.3, 5.9, 11.2, 5.9)
    add_arrow(ax, 10.3, 4.5, 11.2, 4.5)
    add_arrow(ax, 14.1, 4.5, 14.6, 4.5)

    # Teacher branch links
    add_arrow(ax, 6.9, 8.5, 7.4, 8.5)
    add_arrow(ax, 8.85, 5.0, 8.85, 7.8, text="D_vda")
    add_arrow(ax, 10.3, 8.5, 11.2, 8.5)

    # Distillation links
    add_arrow(ax, 12.65, 6.40, 12.65, 8.0, text="(s_hat,t_hat)")
    add_arrow(ax, 14.1, 8.5, 14.6, 8.5)
    add_arrow(ax, 15.55, 8.0, 12.65, 6.40, dashed=True, text="backprop", color="#2E7D57")

    out_dir = Path("/media/a1/16THDD/YJY/VDA_Absolute_Distillation/artifacts/architecture")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "vda_kd_architecture_paper_simple_clean.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(str(out_png))


if __name__ == "__main__":
    main()
