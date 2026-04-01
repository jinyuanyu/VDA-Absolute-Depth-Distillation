# VDA Absolute Distillation

本项目将 Video Depth Anything (VDA) 从“视频相对深度”改造为“单图绝对深度”
方案，核心做法是:

1. 保留 VDA 的相对几何表征能力与边缘质量。
2. 使用 Depth Pro 作为绝对尺度教师。
3. 用轻量 MLP 预测每帧仿射参数 `(s, t)`，得到 `D_abs = s * D_vda + t`。

## 当前落地范围

- 数据目录: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini`
- 已接入相机: `cam00`, `cam01`, `cam04`, `cam05`, `cam19`, `cam20`
- 黑名单: `cam02` (空目录，强制跳过)
- VDA 输入: `.jpg`
- VDA 原始输出: `.npy` (相对深度, `960 x 1280`)
- Depth Pro 输出: `.npz` 键 `depth` (绝对深度, `1014 x 1352`)

## 关键文档

- 规则与边界: `PROJECT_CHARTER_AND_GUIDELINES.md`
- 本次数值/shape 全量剖面: `TECHNICAL_STATUS_2026-03-25.md`
- 数值统计原始 JSON: `runs/run_cam00_01_04_05_19_20/pipeline_numeric_profile_2026-03-25.json`

## 代码结构

- `configs/distill_config.yaml`
- `data_prep/01_spatial_align.py`
- `data_prep/02_extract_scale_labels.py`
- `models/modified_vda.py`
- `models/scale_mlp_head.py`
- `core_engine/dataset.py`
- `core_engine/train_distill.py`
- `inference_abs_vda.py`

## 流程总览

1. `01_spatial_align.py`:
   将 VDA `960 x 1280` 双线性上采样到 Depth Pro 分辨率 `1014 x 1352`。
2. `02_extract_scale_labels.py`:
   先按 `depth_mapping.mode` 对 VDA 深度做变换 (当前默认 `reciprocal_linear`)，
   再用最小二乘求离线伪标签 `(s*, t*)`，并输出 `labels/scale_labels.{json,csv}`。
3. `train_distill.py`:
   冻结 VDA 主干与解码头，仅训练 `ScaleMLPHead`。
4. `inference_abs_vda.py`:
   单图前向得到 `D_vda`，先按 `depth_mapping.mode` 变换，再用 `MLP` 预测
   `(s_hat, t_hat)`，输出绝对深度。

## 当前训练快照

- 运行目录: `runs/run_cam00_01_04_05_19_20`
- 当前最优: `best.pt`, `epoch=6`
- `best_val_loss=0.0028898563541588373`

## 一句话结论

当前版本已经可稳定输出场景内绝对深度，但它是“预训练视觉表征 + 场景内尺度映射”
组合，尚不能证明跨场景泛化，部署前必须做跨场景验证。
