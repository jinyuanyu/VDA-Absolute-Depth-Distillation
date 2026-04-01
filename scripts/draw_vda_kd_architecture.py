from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc, ec="#5b5b5b", lw=1.2, fs=10, weight="normal"):
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


def add_badge(ax, cx, cy, txt, color):
    c = Circle((cx, cy), 0.18, facecolor=color, edgecolor="white", linewidth=1.0)
    ax.add_patch(c)
    ax.text(cx, cy, txt, color="white", ha="center", va="center", fontsize=7, weight="bold")


def add_arrow(ax, x1, y1, x2, y2, dashed=False, text=None, text_pos=0.5):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.3,
        linestyle="--" if dashed else "-",
        color="#4d4d4d",
    )
    ax.add_patch(arr)
    if text:
        tx = x1 + (x2 - x1) * text_pos
        ty = y1 + (y2 - y1) * text_pos
        ax.text(
            tx,
            ty,
            text,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"),
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 18.2)
    ax.axis("off")

    # Palette inspired by referenced TeX style
    input_fill = "#DCEEFF"
    shared_fill = "#EEF1F4"
    student_fill = "#CFE8FF"
    teacher_fill = "#FFE7C2"
    fusion_fill = "#D9F5E5"
    left_band = "#EEF6FF"
    right_band = "#FFF5E7"
    bottom_band = "#EAF8F0"

    # Background bands
    add_box(ax, 0.5, 2.0, 5.2, 14.7, "", left_band, ec=left_band, lw=0.0)
    add_box(ax, 7.1, 6.3, 5.8, 10.4, "", right_band, ec=right_band, lw=0.0)
    add_box(ax, 7.1, 2.0, 5.8, 3.8, "", bottom_band, ec=bottom_band, lw=0.0)

    ax.text(0.6, 17.55, "VDA Absolute Depth Distillation", fontsize=20, weight="bold")
    ax.text(
        0.6,
        17.05,
        "Reference-style vertical architecture: teacher pseudo-labels + frozen student + scale head",
        fontsize=10,
        color="#666666",
    )

    ax.text(0.8, 16.4, "Main student chain", fontsize=11, weight="bold", color="#2B5E85")
    ax.text(7.35, 16.4, "Teacher / label branch", fontsize=11, weight="bold", color="#A86519")
    ax.text(7.35, 5.45, "Distillation & inference", fontsize=11, weight="bold", color="#2E7D57")

    # Left column (student)
    b_in = add_box(ax, 1.0, 14.7, 4.2, 1.35, "Input RGB image", input_fill, fs=10)
    add_badge(ax, 0.78, 15.95, "IN", "#4C90E8")

    b_pre = add_box(
        ax,
        1.0,
        12.45,
        4.2,
        1.35,
        "Single-image preprocess\nmax_res + ensure_multiple_of=14",
        shared_fill,
        fs=9,
    )
    add_badge(ax, 0.78, 13.70, "PP", "#7E8B96")

    b_vda = add_box(
        ax,
        1.0,
        10.15,
        4.2,
        1.55,
        "Frozen VDA Encoder + Decoder\n(T=1, no sliding window)",
        student_fill,
        fs=9,
    )
    add_badge(ax, 0.78, 11.45, "NN", "#3D78C5")

    b_feat = add_box(
        ax,
        1.0,
        7.95,
        4.2,
        1.35,
        "Global feature\nshape: [B, 1024]",
        student_fill,
        fs=9,
    )
    add_badge(ax, 0.78, 9.20, "GF", "#3D78C5")

    b_rel = add_box(
        ax,
        1.0,
        5.75,
        4.2,
        1.35,
        "Relative depth D_vda\n(typically 960x1280)",
        student_fill,
        fs=9,
    )
    add_badge(ax, 0.78, 7.00, "RD", "#3D78C5")

    # Right column (teacher / labels)
    b_tch = add_box(
        ax,
        8.0,
        14.7,
        4.3,
        1.35,
        "Depth-Pro teacher depth\n(npz['depth'])",
        teacher_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 15.95, "TD", "#E6A445")

    b_map = add_box(
        ax,
        8.0,
        12.45,
        4.3,
        1.45,
        "Spatial align + mapping\nresize(D_vda) + phi(d)=1/max(d,eps)",
        teacher_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 13.75, "MP", "#E6A445")

    b_lsq = add_box(
        ax,
        8.0,
        10.10,
        4.3,
        1.45,
        "Least-squares pseudo labels\nsolve (s*, t*) per frame",
        teacher_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 11.40, "LS", "#E6A445")

    b_bank = add_box(
        ax,
        8.0,
        7.90,
        4.3,
        1.35,
        "Offline label bank\nJSON/CSV: (image_path, s*, t*)",
        teacher_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 9.15, "LB", "#E6A445")

    # Bottom right (distillation + inference)
    b_head = add_box(
        ax,
        8.0,
        5.25,
        4.3,
        1.45,
        "Scale MLP Head\npredict (s_hat, t_hat)",
        fusion_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 6.55, "SH", "#4DAA72")

    b_loss = add_box(
        ax,
        8.0,
        3.35,
        4.3,
        1.35,
        "Distill loss\nMSE(s_hat,s*) + MSE(t_hat,t*)",
        fusion_fill,
        fs=9,
    )
    add_badge(ax, 7.78, 4.55, "L", "#4DAA72")

    b_inf = add_box(
        ax,
        4.7,
        1.10,
        4.4,
        1.55,
        "Inference output\nD_abs = s_hat*phi(D_vda)+t_hat\n(+ clamp_min_depth)",
        "#E6F6EC",
        fs=9,
    )
    add_badge(ax, 4.45, 2.45, "OUT", "#4DAA72")

    # Arrows: left chain
    add_arrow(ax, 3.1, 14.7, 3.1, 13.8)
    add_arrow(ax, 3.1, 12.45, 3.1, 11.7)
    add_arrow(ax, 3.1, 10.15, 3.1, 9.3)
    add_arrow(ax, 3.1, 7.95, 3.1, 7.1)

    # Teacher/label branch
    add_arrow(ax, 10.15, 14.7, 10.15, 13.9)
    add_arrow(ax, 10.15, 12.45, 10.15, 11.55)
    add_arrow(ax, 10.15, 10.10, 10.15, 9.25)

    # Cross-branch connections
    add_arrow(ax, 5.2, 6.45, 8.0, 13.2, text="D_vda", text_pos=0.35)
    add_arrow(ax, 5.2, 8.6, 8.0, 6.0, text="global feature", text_pos=0.45)
    add_arrow(ax, 10.15, 7.90, 10.15, 6.7, text="(s*,t*)", text_pos=0.45)
    add_arrow(ax, 10.15, 5.25, 10.15, 4.7)
    add_arrow(ax, 8.0, 4.0, 5.8, 2.65, text="train head", text_pos=0.45)
    add_arrow(ax, 7.9, 5.95, 6.9, 2.65, text="s_hat,t_hat", text_pos=0.45)
    add_arrow(ax, 5.0, 6.2, 5.8, 2.65, text="phi(D_vda)", text_pos=0.60)

    # Update loop to scale head
    add_arrow(ax, 10.0, 3.35, 10.0, 5.25, dashed=True, text="backprop", text_pos=0.45)

    out_dir = Path("/media/a1/16THDD/YJY/VDA_Absolute_Distillation/artifacts/architecture")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "vda_kd_architecture.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(str(out_png))


if __name__ == "__main__":
    main()
