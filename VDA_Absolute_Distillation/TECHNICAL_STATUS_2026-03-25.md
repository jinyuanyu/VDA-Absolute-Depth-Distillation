# 技术状态与数据流说明 (2026-03-25)

本文档给出当前版本的真实输入来源、变量 shape 变化、数值范围、拟合机制、泛化判断与风险结论。

统计来源:

- `runs/run_cam00_01_04_05_19_20/pipeline_numeric_profile_2026-03-25.json`
- `runs/run_cam00_01_04_05_19_20/eval_cam19_cam20_ep5.json`

## 1. 输入来源与样本覆盖

- 数据根目录: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini`
- 相机: `cam00`, `cam01`, `cam04`, `cam05`, `cam19`, `cam20`
- 每个相机 300 帧:
  - 图像: `images/camXX/*.jpg`
  - VDA 深度: `raw_vda_depth/camXX/*.npy`
  - Depth Pro 深度: `raw_depth_pro_depth/camXX/*.npz`
- 总样本数: 1800
- 黑名单: `cam02` (空目录，必须跳过)

## 2. 变量 shape 的变化过程

以 `cam00/0000.jpg` 为例，当前真实 shape 流如下:

1. 原始图像读取:
   - `input_rgb_shape = [1014, 1352, 3]`
2. 进入 `SingleImageVDA.prepare_batch`:
   - 先受 `max_res=1280` 约束，图像缩放为 `960 x 1280`
   - 再经 `input_size=518 + ensure_multiple_of=14`，网络输入 tensor 为
     `prepared_batch_shape = [1, 3, 518, 686]`
3. VDA 特征:
   - `global_feature_shape = [1, 1024]`
4. VDA 相对深度:
   - 网络内部预测 `relative_depth_pre_upsample_shape = [1, 518, 686]`
   - 在 `infer_single_image` 中回采样为 `relative_depth_final_shape = [960, 1280]`
5. 离线标签拟合:
   - 再将 VDA `960 x 1280` 上采样到 Depth Pro `1014 x 1352`
   - 与 `depth_pro` 逐像素拟合 `(s*, t*)`

要点:

- 训练阶段输入 `image_path`，每张图会重新经过 VDA 前处理抽全局特征。
- 标签阶段的分辨率对齐与推理阶段的分辨率链路不是同一步，但最终都围绕
  `960 x 1280 <-> 1014 x 1352` 变换。

## 3. 数值范围 (当前真实统计)

### 3.1 VDA 原始相对深度 (raw_vda_depth)

跨 6 相机总体统计:

- 全局最小值: `57.5794`
- 全局最大值: `1786.8192`
- 各帧 p1 的均值: `127.0651`
- 各帧 p99 的均值: `1467.0531`

解释:

- 这是一组“相对深度标量”，不是物理米制深度。
- 数值量纲与尺度范围明显不同于 Depth Pro。

### 3.2 Depth Pro 绝对深度 (raw_depth_pro_depth, key=depth)

跨 6 相机总体统计:

- 全局最小值: `0.4857`
- 全局最大值: `4.8345`
- 各帧 p1 的均值: `0.7087`
- 各帧 p99 的均值: `2.6813`

解释:

- 这组值处于“米级”范围，可作为绝对尺度教师。

## 4. 在新算法中如何利用 VDA 原始输出

### 4.1 标签提取阶段 (线性)

- 输入: `D_vda_up` (上采样后) 与 `D_depth_pro`
- 输出: 每帧 `(s*, t*)`
- 形式: 线性仿射拟合，目标是最小二乘

`D_depth_pro ~= s * D_vda_up + t`

当前整体标签统计 (6 相机平均):

- `scale_mean_across_cams = -0.0014148`
- `shift_mean_across_cams = 2.4103`
- `residual_mse_mean_across_cams = 0.04352`

注意:

- `scale` 均值为负，说明当前 VDA 相对深度方向与教师定义方向相反。
- 下游必须使用仿射后的深度，不可把原始 VDA 数值直接当绝对深度。

### 4.2 蒸馏训练阶段 (非线性映射)

- 训练对象: `ScaleMLPHead` (3 层 MLP, ReLU)
- 冻结: VDA Encoder + Decoder
- 监督: `(s*, t*)`
- 本质: 学的是 `feature -> (s, t)` 的非线性回归，不是直接回归整张深度图

当前训练快照:

- 运行目录: `runs/run_cam00_01_04_05_19_20`
- 最优 checkpoint: `best.pt`, `epoch=6`
- `best_val = 0.0028898563541588373`

### 4.3 推理阶段

1. VDA 单图输出 `D_vda`
2. MLP 预测 `(s_hat, t_hat)`
3. 生成 `D_abs = s_hat * D_vda + t_hat`
4. 按配置执行最小值截断 (`clamp_min_depth`)

## 5. 当前效果观察

基于 `best.pt` 的场景内评估 (`cam19/cam20`):

- `cam19`: `mse_mean = 0.1294`
- `cam20`: `mse_mean = 0.2248`

相较早期权重，误差有明显下降，说明 scale head 已经学到“本场景内”的有效尺度校正。

## 6. 线性还是非线性

- `s*, t*` 离线伪标签提取: 线性最小二乘 (闭式解)
- `feature -> (s_hat, t_hat)` 学习器: 非线性 MLP
- `D_abs` 生成: 对像素值应用线性仿射变换

所以整个系统是“非线性预测参数 + 线性深度变换”的组合。

## 7. 是学到通用视觉知识，还是过拟合当前场景

结论:

- 通用视觉知识主要来自“冻结的 VDA 预训练 backbone”。
- 新训练的 MLP 只在当前场景/相机分布上学习尺度映射，具备明显场景依赖风险。
- 当前验证集来自同场景随机拆分，不能证明跨场景泛化。

因此:

- 目前结果更接近“场景内有效的尺度校正器”，而非“已验证的通用绝对深度器”。

## 8. 一直要盯住的核心问题

1. 教师噪声传递: Depth Pro 非真值，FOV 估计误差会被蒸馏继承。
2. 场景泄漏: 同场景随机拆分易高估泛化能力。
3. 方向一致性: `scale` 符号变化意味着 VDA 与教师定义不一致风险。
4. 分辨率链路: `1014x1352 -> 960x1280 -> 518x686 -> 960x1280 -> 1014x1352`
   任一处改动都可能影响拟合分布。
5. 预处理一致性: 均值方差、`ensure_multiple_of=14`、`max_res` 必须固定。
6. 指标解释: 低 val loss 仅代表 `(s, t)` 拟合好，不等价于跨域深度可用。

## 9. 建议的下一步验证

1. 使用“按场景划分”的 train/val/test，而不是按帧随机划分。
2. 增加跨场景测试集，报告 `abs-rel`, `rmse`, `delta` 等指标。
3. 对 `scale`/`shift` 的时间稳定性做曲线检查，定位异常帧并与 FOV 抖动关联分析。
