# Project Charter & Guidelines

本文件定义 VDA 绝对深度蒸馏项目的工程目标、硬约束、边界条件和长期注意事项。

## 1. 项目目标

1. 将 VDA 从“依赖时序窗口的相对深度模型”改造为“单图输入的绝对深度模型”。
2. 保留 VDA 的视觉几何先验与边缘细节，同时引入 Depth Pro 的绝对尺度监督。
3. 通过 VDA 全局特征预测每帧 `(s, t)`，缓解 Depth Pro 在缺少 EXIF 焦距时的
   帧间尺度抖动。

## 2. 约束与边界

### 2.1 数据与目录约束

- 当前允许相机: `cam00`, `cam01`, `cam04`, `cam05`, `cam19`, `cam20`
- 强制跳过: `cam02` (空目录)
- VDA 深度输入格式: `.npy`
- Depth Pro 深度输入格式: `.npz`，键名固定为 `depth`

### 2.2 分辨率与空间对齐约束

- VDA 原始深度分辨率: `960 x 1280`
- Depth Pro 原始深度分辨率: `1014 x 1352`
- 离线拟合 `(s*, t*)` 前，必须先将 VDA 深度双线性上采样到 `1014 x 1352`。

### 2.3 网络与训练约束

- 蒸馏训练必须冻结 VDA ViT Encoder。
- 蒸馏训练必须冻结 VDA Decoder / Head。
- 单图前向必须满足 `ensure_multiple_of = 14` 的 patch 对齐规则。
- 新管线移除原 32 帧窗口与重叠逻辑，固定时间维 `T = 1`。

## 3. 技术路线与数学形式

### 3.1 离线伪标签提取

对每帧求解:

`(s*, t*) = argmin_{s, t} sum_{x,y} (s * phi(D_vda_up(x,y)) + t - D_depth_pro(x,y))^2`

其中 `phi` 由 `depth_mapping.mode` 控制，当前默认:

`phi(d) = 1 / max(d, eps)` (`reciprocal_linear`)

实现采用闭式最小二乘解，属于线性拟合问题。

### 3.2 Scale Head 学习

`(s_hat, t_hat) = MLP(Global_Avg_Pool(VDA_Encoder_Features))`

损失:

`L = lambda_s * ||s_hat - s*||_2^2 + lambda_t * ||t_hat - t*||_2^2`

此阶段中，映射网络是非线性 MLP，但其学习目标是“从全局语义特征到线性仿射参数”。

## 4. 当前落地事实

- 数据规模: 6 个相机 * 300 帧 = 1800 样本
- 标签文件: `labels/scale_labels.json` 与 `labels/scale_labels.csv`
- 训练快照: `runs/run_cam00_01_04_05_19_20/best.pt`，当前最优 `epoch=6`
- 评估快照: `runs/run_cam00_01_04_05_19_20/eval_cam19_cam20_ep5.json`

## 5. 长期必须关注的核心问题

1. 伪标签来自 Depth Pro，不是激光真值，教师噪声会被学生继承。
2. 当前训练与验证均来自同一场景，随机切分无法证明跨场景泛化。
3. `(s, t)` 学到的是场景条件映射，不应等同于“通用尺度恢复能力”。
4. VDA 相对深度与 Depth Pro 方向/尺度定义可能不同，必须使用仿射后结果。
5. 分辨率对齐与插值策略会影响拟合误差，变更时要重算标签并复训。
6. 输入预处理配置是敏感项，均值方差和 patch 对齐规则不能随意改。
7. 如果引入新场景，优先做跨场景留一验证，而不是只看场景内 val loss。

## 6. 交付标准

- 每次改动后至少提供:
  - 标签统计 (`s/t/residual`)。
  - 一组可视化对比图 (`VDA`, `VDA-Absolute`, `Depth-Pro`)。
  - 一份跨相机误差摘要。
- 未通过上述检查，不进入下游系统联调。
