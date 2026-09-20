# Boundary-Aware Lightweight BiSeNetV2 for Off-Road Traversability Segmentation on RK3588
## 当前研究成果与论文写作参考文档

> 状态：截至 Experiment C 完成后的阶段性总结  
> 目标：作为后续 EI 会议论文撰写、实验补充和结果核对的统一参考文档。  
> 注意：本文档将“已完成实验”和“计划/待完成实验”明确区分，未完成内容不应在论文中表述为已有结果。

---

## 1. 当前论文题目

**英文暂定题目：**

**Boundary-Aware Lightweight BiSeNetV2 for Off-Road Traversability Segmentation on RK3588**

可在最终写作阶段根据实际贡献进一步微调，例如：

- **A Lightweight Boundary-Aware BiSeNetV2 for Off-Road Traversability Segmentation on RK3588**
- **Boundary-Aware Lightweight Semantic Segmentation for Off-Road Traversability Perception on RK3588**

目前推荐继续使用第一版标题，等 Experiment D、RK3588 部署和跨数据集结果完成后再最终定题。

---

## 2. 研究背景

非结构化越野环境中的可通行区域感知是移动机器人、无人地面车辆和智能驾驶系统的重要基础能力。

与城市道路相比，越野场景通常具有以下特点：

1. **缺乏规则道路结构。** 越野场景中不存在稳定的车道线、路缘等人工结构，可通行区域常由草地、泥土、碎石等自然地形组成。
2. **类别边界模糊。** 可通行区域与灌木、岩石、水体等不可通行区域之间往往不存在清晰几何边界。
3. **视觉外观变化大。** 不同序列、天气、光照、植被类型和地形条件会带来明显的域差异。
4. **边界错误具有实际意义。** 对机器人路径规划而言，仅获得较高区域 IoU 并不足够。可通行/不可通行交界处的错误可能直接影响规划边界的位置。
5. **部署平台资源受限。** 实际移动机器人通常依赖嵌入式计算平台，因此模型不仅需要较好的精度，还需要较低的参数量、计算量和推理延迟。

BiSeNetV2 本身是一种适合实时语义分割的双分支轻量网络：

- Detail Branch 保留空间细节；
- Semantic Branch 提取高级语义信息；
- Bilateral Guided Aggregation（BGA）融合两类特征。

本研究以 BiSeNetV2 为基础，重点研究：

> **能否通过轻量级边界监督和边界特征细化，在几乎不增加或只少量增加推理成本的前提下，提高越野可通行区域边界质量。**

---

## 3. 研究目标

当前工作的核心目标不是单纯追求最高 mIoU，而是同时考虑：

- 区域级语义分割精度；
- 可通行/不可通行边界质量；
- 模型轻量化程度；
- 嵌入式平台部署效率；
- sequence-disjoint 条件下的泛化能力。

核心研究问题可以归纳为：

### RQ1：显式边界监督是否有效？

在不改变 BiSeNetV2 推理路径的情况下，引入训练期 Boundary Head 和 Boundary Loss，是否能够提升最终 semantic prediction 的边界质量？

对应 **Experiment B**。

### RQ2：轻量边界细化模块是否有效？

在不使用任何 Boundary GT 和 Boundary Loss 的情况下，仅在 BGA 后增加轻量 Boundary Refinement Module，是否能够改善最终预测边界？

对应 **Experiment C**。

### RQ3：两种机制是否互补？

将 Boundary Supervision 和 Boundary Refinement 同时使用，是否能兼顾区域分割性能与边界质量？

对应 **Experiment D（待完成）**。

### RQ4：改进模型能否在 RK3588 上保持实时/近实时运行？

对应后续 ONNX/RKNN FP16/INT8 部署实验。

---

## 4. 数据集与任务定义

### 4.1 主数据集

当前主数据集为 **RUGD（Rugged Outdoor Dataset）**。

原始数据包含大量细粒度越野语义类别。早期实验曾构建四分类版本，但发现将 bush / rock-bed / water 等视觉和几何性质差异较大的区域合并为单一类别会导致明显的 sequence-dependent 分类冲突。

因此最终论文主任务确定为 **三分类 Traversability Segmentation**。

### 4.2 最终三分类定义

| Label | Class | 含义 |
|---:|---|---|
| 0 | sky | 天空 |
| 1 | traversable | 可通行区域 |
| 2 | non_traversable | 不可通行区域 |
| 255 | ignore | 忽略区域 |

三分类方案由原四分类标签进一步合并得到：

- 原 0 → 0
- 原 1 → 1
- 原 2 → 2
- 原 3 → 2
- 255 → 255

### 4.3 数据划分

采用严格 **sequence-disjoint split**，保证训练、验证和测试序列不重叠。

#### Validation sequences

- trail-4
- trail-5

#### Test sequences

- trail
- trail-10
- trail-12
- trail-14

#### Train sequences

- creek
- park-1
- park-2
- park-8
- trail-11
- trail-13
- trail-15
- trail-3
- trail-6
- trail-7
- trail-9
- village

### 4.4 图像数量

| Split | Images |
|---|---:|
| Train | 4894 |
| Validation | 1134 |
| Test | 1408 |
| Total | 7436 |

### 4.5 三分类像素分布

| Split | Sky | Traversable | Non-traversable |
|---|---:|---:|---:|
| Train | 8.85% | 40.31% | 50.85% |
| Validation | 5.50% | 47.76% | 46.74% |
| Test | 5.72% | 46.64% | 47.64% |

新的三分类方案使 traversable / non-traversable 在 Val/Test 上保持较为均衡的像素比例。

---

## 5. 早期四分类实验及其意义

早期 RUGD4 方案：

- 0 sky
- 1 traversable
- 2 non-traversable terrain：bush / rock-bed / water
- 3 obstacle

四分类 baseline 的性能约为：

| Split | SS mIoU |
|---|---:|
| Validation | 65.29% |
| Test | 64.92% |

主要失败类别为原 class 2。

其中 class 2 的内部组成具有明显分布漂移：

### Train

- bush：62.29%
- rock-bed：33.08%
- water：4.63%

### Validation

- bush：91.79%
- rock-bed：5.27%
- water：2.93%

### Test

- bush：96.44%
- rock-bed：0.11%
- water：3.45%

进一步检查确认标签转换本身无错误，train / val / test 转换后的目标像素数量与原始标签完全一致。

因此四分类低性能更可能来源于：

> **不合理的 ontology 合并和 sequence-dependent 类内分布差异，而不是代码或标签转换错误。**

将四分类预测结果离线合并为三分类后，mIoU 可恢复至约 81% 左右；重新训练的三分类 BiSeNetV2 也得到相近结果。

这支持最终采用三分类作为论文主任务。

---

## 6. Baseline 与统一训练设置

### 6.1 Baseline

**BiSeNetV2**

官方 Cityscapes 预训练权重：

`model_final_v2_city.pth`

Cityscapes 19 类输出层与 RUGD3 三分类不兼容，因此载入预训练权重时：

- compatible tensors loaded：373
- incompatible output tensors skipped：10

跳过的主要是：

- main segmentation head output
- aux2 output
- aux3 output
- aux4 output
- aux5_4 output

### 6.2 正式训练设置

当前 A/B/C 均保持相同主要训练协议：

| Item | Setting |
|---|---|
| GPU | NVIDIA RTX 5090 |
| Dataset | RUGD3 |
| Classes | 3 |
| Crop size | 512 × 640 |
| Batch size | 16 |
| Seed | 123 |
| Initial LR | 2.5e-3 |
| Weight decay | 5e-4 |
| Warmup | 1000 iterations |
| Max iterations | 80000 |
| Save interval | 5000 |
| Mixed precision | FP16 |
| Pretrained model | Cityscapes BiSeNetV2 |

### 6.3 Checkpoint 选择规则

所有正式实验统一采用：

> **Validation single-scale（SS）mIoU 最大的 checkpoint 作为 best checkpoint。**

以下指标均不用于 checkpoint selection：

- Test mIoU
- Boundary F1
- multi-scale mIoU
- crop mIoU

Test 集只在 best checkpoint 冻结后进行一次正式评估。

---

# 7. Experiment A：Baseline BiSeNetV2

## 7.1 目的

建立不含任何 boundary-aware 机制的标准 BiSeNetV2 baseline。

## 7.2 Best checkpoint

Validation checkpoint 对比：

| Iter | Val SS mIoU |
|---:|---:|
| 40000 | 79.7002% |
| 45000 | 81.2152% |
| 50000 | 82.1069% |
| 55000 | 82.0893% |
| **60000** | **82.7955%** |
| 65000 | 81.2978% |
| 70000 | 81.2318% |
| 75000 | 81.4309% |
| 80000 | 81.7008% |

因此：

**Experiment A best = 60000 iterations**

## 7.3 Validation

| Metric | Value |
|---|---:|
| mIoU | **82.7955%** |
| Macro F1 | 89.9870% |
| Sky IoU | 64.2915% |
| Traversable IoU | 94.0219% |
| Non-traversable IoU | 90.0730% |
| BF1-All@3 | 59.0312% |
| BF1-Trav@3 | 49.5498% |

Validation BF1-Trav：

- Precision：58.8873%
- Recall：42.7682%
- F1：49.5498%

## 7.4 Test

| Metric | Value |
|---|---:|
| mIoU | **81.1822%** |
| Macro F1 | 89.4825% |
| Sky IoU | 72.5399% |
| Traversable IoU | 86.6151% |
| Non-traversable IoU | 84.3915% |
| BF1-All@3 | 51.0820% |
| BF1-Trav@3 | 31.5507% |

Test BF1-Trav：

- Precision：37.8311%
- Recall：27.0587%
- F1：31.5507%

核心 Test confusion：

- traversable → non-traversable：8.46%
- non-traversable → traversable：5.45%

---

# 8. Experiment B：Boundary Supervision Only

## 8.1 设计目的

验证：

> **不增加推理路径计算量，仅通过训练期显式边界监督，是否能够改善最终 semantic prediction 的边界质量。**

### 结构

训练：

`BGA feature → segmentation head`

同时：

`BGA feature → training-only boundary head`

Boundary Head 仅在 `aux_mode="train"` 时执行。

Eval / deployment 时：

- Boundary Head 完全不执行；
- semantic inference path 与 baseline 相同；
- 因此 deployment inference overhead 理论上为 0。

已通过 forward hook 验证：

- train boundary calls = 1
- eval boundary calls = 0

## 8.2 Boundary GT

Boundary GT 来自 semantic label 的类别变化。

训练时监督所有 semantic boundaries：

- sky ↔ traversable
- sky ↔ non-traversable
- traversable ↔ non-traversable

Boundary width：**3 pixels**

第一批有效区域 boundary positive ratio：**4.6335%**

## 8.3 Boundary Loss

使用：**BCE + Dice**

总损失：

`L = L_main + ΣL_aux + λ_boundary × L_boundary`

其中：**λ_boundary = 0.2**

Boundary Loss 使用 FP32 计算以提高数值稳定性。

## 8.4 Best checkpoint

按 Validation SS mIoU：

**Experiment B best = 75000 iterations**

Val SS mIoU：**83.1244%**

## 8.5 Validation

| Metric | A | B | Δ(B-A) |
|---|---:|---:|---:|
| mIoU | 82.7955% | **83.1244%** | **+0.3289 pp** |
| Macro F1 | 89.9870% | **90.2139%** | +0.2269 pp |
| Sky IoU | 64.2915% | **65.0331%** | +0.7416 pp |
| Traversable IoU | 94.0219% | **94.1263%** | +0.1044 pp |
| Non-traversable IoU | 90.0730% | **90.2139%** | +0.1409 pp |
| BF1-All@3 | 59.0312% | **60.4139%** | **+1.3827 pp** |
| BF1-Trav@3 | 49.5498% | **52.3167%** | **+2.7669 pp** |

B Validation BF1-Trav：

- Precision：60.6024%
- Recall：46.0241%
- F1：52.3167%

### Boundary tolerance robustness

| Tolerance | A BF1-Trav | B BF1-Trav | Δ |
|---:|---:|---:|---:|
| 2 px | 42.0836% | **44.8319%** | +2.7483 pp |
| 3 px | 49.5498% | **52.3167%** | +2.7669 pp |
| 5 px | 59.6229% | **62.0803%** | +2.4574 pp |

B 在 2/3/5 px 三种 tolerance 下均稳定优于 A。

## 8.6 Test

| Metric | A | B | Δ(B-A) |
|---|---:|---:|---:|
| mIoU | 81.1822% | **81.2181%** | +0.0359 pp |
| Macro F1 | 89.4825% | **89.5183%** | +0.0358 pp |
| Sky IoU | 72.5399% | **73.0330%** | +0.4931 pp |
| Traversable IoU | **86.6151%** | 86.3079% | -0.3072 pp |
| Non-traversable IoU | **84.3915%** | 84.3133% | -0.0782 pp |
| BF1-All@3 | 51.0820% | **53.1307%** | **+2.0487 pp** |
| BF1-Trav@3 | 31.5507% | **34.1914%** | **+2.6407 pp** |

B Test BF1-Trav：

- Precision：42.3642%
- Recall：28.6620%
- F1：34.1914%

Test confusion：

- traversable → non-traversable：9.11%
- non-traversable → traversable：5.07%

相比 A：

- non-traversable → traversable：5.45% → 5.07%
- traversable → non-traversable：8.46% → 9.11%

说明 B 的预测略趋于保守。

## 8.7 Experiment B 当前结论

Boundary Supervision：

- 对 Test mIoU 几乎无影响；
- 对 Validation/Test Boundary F1 都有稳定提升；
- BF1-Trav 的提升比 mIoU 更显著；
- 推理阶段无需 Boundary Head，因此不增加部署计算路径。

可以认为：

> **显式训练期 boundary supervision 能够改善共享特征对 traversability boundary 的建模能力，而不需要额外的推理分支。**

---

# 9. Experiment C：Boundary Refinement Only

## 9.1 设计目的

验证：

> **在完全不使用 Boundary GT 和 Boundary Loss 的情况下，仅通过轻量的推理期 Boundary Refinement Module 是否可以改善 boundary quality。**

## 9.2 模块位置

结构：

`Detail Branch + Semantic Branch → BGA → LBRM → Segmentation Head`

LBRM 位于：**BGA 后、主 segmentation head 前**

实际 feature shape 已验证：

- LBRM input：`1 × 128 × 64 × 80`
- LBRM output：`1 × 128 × 64 × 80`

即模块工作于约 H/8 × W/8 特征空间，而非原始高分辨率图像空间。

## 9.3 Lightweight Boundary Refinement Module（LBRM）

核心思想：

1. 对 BGA feature 做局部平均；
2. 使用 `F - AvgPool(F)` 产生 local contrast；
3. 从 local contrast 学习 spatial boundary gate；
4. 使用轻量 bottleneck + depthwise convolution 生成 refinement feature；
5. 通过 gated residual 方式增强原始 feature。

形式可表示为：

`C = F - AvgPool(F)`

`G = sigmoid(phi(C))`

`ΔF = psi(F)`

`F' = ReLU(F + G ⊙ ΔF)`

LBRM 无任何显式 boundary label。

## 9.4 模块复杂度

实际参数量：

| Model | Parameters |
|---|---:|
| A BiSeNetV2 | 5,193,263 |
| C BiSeNetV2 + LBRM | 5,204,352 |
| Added | **11,089** |

即：

- +0.011089 M
- **+0.2135%**

### RTX 5090 forward benchmark

Input：`1 × 3 × 576 × 704`  
Warmup：100  
Runs：500

| Model | Latency | FPS |
|---|---:|---:|
| A | 2.644 ms | 378.17 |
| C | 2.931 ms | 341.24 |
| Δ | **+0.286 ms** | |
| Relative latency increase | **+10.82%** | |

注：参数增加很少并不意味着 latency 线性增加。Pool、sigmoid、depthwise convolution、element-wise operations 仍具有 kernel launch 和 memory-access 开销。

最终“轻量性”需要以 RK3588/RKNN 实测为主要依据。

## 9.5 Best checkpoint

**已通过 Validation SS mIoU 选择并冻结 `best_model.pth`。**

**Experiment C best = 70000 iterations**

对应 Validation SS mIoU：**83.0483%**

## 9.6 Validation

| Metric | A | B | C |
|---|---:|---:|---:|
| mIoU | 82.7955% | **83.1244%** | 83.0483% |
| Macro F1 | 89.9870% | **90.2139%** | 90.1505% |
| Sky IoU | 64.2915% | **65.0331%** | 64.6944% |
| Traversable IoU | 94.0219% | 94.1263% | **94.2101%** |
| Non-traversable IoU | 90.0730% | 90.2139% | **90.2404%** |
| BF1-All@3 | 59.0312% | **60.4139%** | 60.3212% |
| BF1-Trav@3 | 49.5498% | 52.3167% | **53.0774%** |

C 相比 A：

- mIoU：+0.2528 pp
- BF1-All@3：+1.2900 pp
- BF1-Trav@3：**+3.5276 pp**

C Validation BF1-Trav：

- Precision：61.4053%
- Recall：46.7387%
- F1：53.0774%

### Boundary tolerance robustness

| Tolerance | A | B | C |
|---:|---:|---:|---:|
| 2 px | 42.0836% | 44.8319% | **45.5328%** |
| 3 px | 49.5498% | 52.3167% | **53.0774%** |
| 5 px | 59.6229% | 62.0803% | **62.8811%** |

三个 tolerance 下均：**C > B > A**

说明 C 的 boundary improvement 并非由特定 tolerance 设置偶然造成。

## 9.7 Test

| Metric | A | B | C |
|---|---:|---:|---:|
| mIoU | 81.1822% | **81.2181%** | 80.9800% |
| Macro F1 | 89.4825% | **89.5183%** | 89.3804% |
| Sky IoU | 72.5399% | 73.0330% | **73.0575%** |
| Traversable IoU | **86.6151%** | 86.3079% | 85.8728% |
| Non-traversable IoU | **84.3915%** | 84.3133% | 84.0098% |
| BF1-All@3 | 51.0820% | 53.1307% | **53.1454%** |
| BF1-Trav@3 | 31.5507% | 34.1914% | **35.6351%** |

C 相比 A：

- Test mIoU：**-0.2022 pp**
- BF1-All@3：+2.0634 pp
- BF1-Trav@3：**+4.0844 pp**

C Test BF1-Trav：

- Precision：43.6625%
- Recall：30.1010%
- F1：35.6351%

### Test confusion

C：

- traversable → non-traversable：9.64%
- non-traversable → traversable：4.99%

A/B/C 对比：

| Error | A | B | C |
|---|---:|---:|---:|
| traversable → non-traversable | 8.46% | 9.11% | 9.64% |
| non-traversable → traversable | 5.45% | 5.07% | **4.99%** |

C 的预测表现出更明显的保守倾向：

- 更少将不可通行区域错误识别为可通行；
- 但更多将实际可通行区域预测为不可通行。

这一结果可以讨论为风险相关的错误类型变化，但当前不能将其表述为完整“安全性证明”。

## 9.8 Experiment C 当前结论

Experiment C 的结果表现为明显的 accuracy-boundary trade-off：

### Validation

- mIoU：+0.25 pp
- BF1-Trav@3：+3.53 pp

### Test

- mIoU：-0.20 pp
- BF1-Trav@3：+4.08 pp

因此：

> **LBRM 对 traversability boundary 的改善具有较强的 sequence-disjoint 泛化能力，但在 Test 集上存在轻微区域级 mIoU 损失。**

这正是 Experiment D 需要进一步解决的问题。

---

# 10. 当前主消融实验汇总

| Exp | Boundary Supervision | Boundary Refinement | Val mIoU | Test mIoU | Val BF1-Trav@3 | Test BF1-Trav@3 |
|---|---|---|---:|---:|---:|---:|
| A | × | × | 82.80% | 81.18% | 49.55% | 31.55% |
| B | ✓ | × | **83.12%** | **81.22%** | 52.32% | 34.19% |
| C | × | ✓ | 83.05% | 80.98% | **53.08%** | **35.64%** |
| D | ✓ | ✓ | 待完成 | 待完成 | 待完成 | 待完成 |

当前结果已经支持两个独立结论：

1. **Boundary Supervision 独立有效。**
2. **Boundary Refinement 独立有效。**

两者的收益形式有所不同：

- B 更偏向保持区域精度并提升边界质量；
- C 获得更大的 Boundary F1 提升，但在 Test 上轻微牺牲 mIoU。

因此 Experiment D 有明确实验动机。

---

# 11. Experiment D：Joint Boundary Supervision + Refinement（待完成）

建议结构：

`Detail + Semantic → BGA → LBRM → refined feature → semantic head`

同时训练期：

`refined/BGA feature → Boundary Head → Boundary Loss`

应保持 B/C 已固定的超参数，避免引入新变量：

### Boundary Supervision

- boundary width = 3
- loss = BCE + Dice
- λ_boundary = 0.2

### LBRM

- channels = 128
- bottleneck channels = 32
- gate channels = 16

### 实验控制

Experiment D 应：

- 从同一个 `model_final_v2_city.pth` 开始；
- 使用同样 seed=123；
- 80k iterations；
- 每 5k 保存；
- 按 Val SS mIoU 选 best；
- 不根据 Test 调整参数。

理想但非强制目标：

- Test mIoU 接近或超过 B；
- BF1-Trav 接近或超过 C；
- 实现更好的 accuracy-boundary-efficiency 综合折中。

---

# 12. Boundary F1 定义

当前 Boundary F1 并非 Boundary Head 自身输出的准确率，而是：

`semantic logits → argmax segmentation → semantic boundary extraction → tolerance matching`

因此 A/B/C/D 均可以使用同一个评价方法，保证公平比较。

当前报告两个指标：

### BF1-All

所有有效 semantic class boundary。

### BF1-Trav

仅考虑：**traversable ↔ non-traversable**

这是论文主 Boundary 指标。

推荐主结果使用：**BF1-Trav@3**

同时以 @2 / @5 作为 robustness analysis。

---

# 13. Sequence-disjoint 泛化现象

A/B/C 都表现出：

- Test mIoU 相对 Val 下降有限；
- Test BF1-Trav 相对 Val 明显下降。

例如：

### A

- Val BF1-Trav@3：49.55%
- Test BF1-Trav@3：31.55%

### B

- Val：52.32%
- Test：34.19%

### C

- Val：53.08%
- Test：35.64%

说明：

> **Boundary localization 相比区域级语义分割对 unseen sequences 的 domain shift 更敏感。**

这一现象可以成为论文的重要讨论点，并进一步支撑研究 boundary-aware perception 的必要性。

---

# 14. 当前论文可以主张的阶段性贡献

在 Experiment D 和部署实验完成前，当前结果已经可以支持以下贡献方向：

### Contribution 1：面向越野 traversability 的三分类任务重构

通过早期四分类实验和类别内部统计，发现将视觉/几何性质差异较大的 bush、water、rock-bed 等合并为单一中间类别会产生明显 sequence-dependent 冲突。

最终采用 sky / traversable / non-traversable 三分类设计，并在 strict sequence-disjoint split 下获得稳定 baseline。

### Contribution 2：Training-only Boundary Supervision

设计训练期 Boundary Head 和 BCE+Dice Boundary Loss：

- 不参与 inference；
- Test mIoU 基本保持；
- Test BF1-Trav@3 提升约 2.64 pp。

### Contribution 3：Lightweight Boundary Refinement

在 BGA 后加入约 0.011M 参数的 LBRM：

- 参数仅增加约 0.21%；
- Val BF1-Trav@3 提升约 3.53 pp；
- Test BF1-Trav@3 提升约 4.08 pp；
- Boundary improvements 在 2/3/5 px tolerance 上均稳定存在。

### Contribution 4：嵌入式部署（待验证）

目标是在 RK3588 上完成：

- ONNX export
- RKNN conversion
- FP16
- INT8
- latency/FPS
- memory
- accuracy degradation

只有完成后，才能作为正式贡献写入摘要/结论。

---

# 15. 当前论文结论草案

当前实验结果初步表明：

1. 经过合理的 traversability ontology 重构后，BiSeNetV2 可以在 RUGD strict sequence-disjoint split 上获得约 81–83% mIoU。
2. 单纯使用训练期 Boundary Supervision 即可稳定提高 traversability boundary quality，而无需增加部署推理路径。
3. LBRM 通过极少的额外参数能够进一步增强 traversable/non-traversable 边界定位，且提升能够从 Validation 泛化到 unseen Test sequences。
4. Boundary F1 的跨序列下降明显大于区域 mIoU，说明边界预测是越野场景中更具挑战性的泛化问题。
5. C 在 Test 上呈现轻微的区域精度与边界质量 trade-off，说明后续需要通过联合 Boundary Supervision + Refinement 的 Experiment D 寻找更优平衡。

---

# 16. 推荐论文整体结构

## Abstract

最终完成 D、RK3588 和 cross-dataset 后再写。

摘要至少包含：

- 越野边界问题；
- 提出的 boundary-aware lightweight BiSeNetV2；
- RUGD sequence-disjoint 实验；
- Boundary F1 和 mIoU 改善；
- RK3588 latency/FPS。

## 1. Introduction

建议逻辑：

1. 越野 traversability perception 的重要性；
2. 语义区域正确但边界错误仍可能影响路径规划；
3. 高精度模型难以部署到嵌入式平台；
4. BiSeNetV2 是适合实时应用的基础；
5. 提出 training-time boundary supervision + lightweight boundary refinement；
6. 列出贡献。

## 2. Related Work

建议分为：

- Semantic segmentation
- Lightweight real-time segmentation
- Boundary-aware segmentation
- Off-road traversability perception

## 3. Method

### 3.1 Baseline BiSeNetV2
### 3.2 Traversability Ontology
### 3.3 Boundary Supervision
### 3.4 Lightweight Boundary Refinement Module
### 3.5 Joint Objective

## 4. Experiments

### 4.1 Dataset and Split
### 4.2 Implementation Details
### 4.3 Metrics
### 4.4 Ablation Study
### 4.5 Boundary Analysis
### 4.6 Efficiency Analysis
### 4.7 RK3588 Deployment
### 4.8 Cross-dataset Evaluation（如完成）

## 5. Discussion

重点讨论：

- ontology；
- boundary/generalization；
- conservative error tendency；
- mIoU vs BF1；
- lightweight trade-off；
- limitations。

## 6. Conclusion

等全部实验完成后再定稿。

---

# 17. 接下来必须完成的实验

## 高优先级

### 1. Experiment D

这是当前论文完整方法最关键的缺口。

### 2. RK3588 部署

至少需要：

| Model | Params | ONNX/RKNN | Precision | Latency | FPS |
|---|---:|---|---|---:|---:|
| A | | | FP16 | | |
| B | | | FP16 | | |
| C | | | FP16 | | |
| D | | | FP16 | | |

推荐再补 D：

- INT8 latency
- INT8 accuracy

### 3. Boundary-aware qualitative visualization

至少选若干 Test 图：

- 原图
- GT
- A prediction
- B prediction
- C prediction
- D prediction
- boundary overlay / error map

应重点选择：

- vegetation boundary；
- trail boundary；
- shadow / illumination；
- rock / grass transition；
- distant or thin obstacles。

## 中优先级

### 4. Cross-dataset validation

原计划使用：**RELLIS-3D**

建议将其重映射到相同三分类 ontology，作为 external validation。

### 5. FLOPs / MACs

至少给：A、C/D。

### 6. Model size / runtime memory

用于补充部署效率。

---

# 18. 当前仍需用户补充或确认的数据

为了后面能够直接写成完整论文，建议补充以下内容。

## 必需

### 1. Experiment C 的 BEST ITER

已确认：

- **BEST ITER = 70000**
- Val SS mIoU = 83.0483%
- `best_model.pth` 已冻结

### 2. Experiment D 全部结果

待训练后补：

- checkpoint curve
- best iteration
- Val class IoU / mIoU / F1
- Val BF1
- Test class IoU / mIoU / F1
- Test BF1
- confusion
- tolerance robustness

### 3. RK3588 实测

需要：

- RK3588 板卡/系统信息
- RKNN Toolkit / Runtime 版本
- inference input resolution
- FP16 latency/FPS
- INT8 latency/FPS
- quantization dataset
- accuracy before/after quantization
- CPU/NPU utilization（如可获得）

## 强烈推荐

### 4. FLOPs/MACs

A/B/C/D。

### 5. C 与未来 D 的正式模型文件大小

例如 `.pth / .onnx / .rknn`。

### 6. Qualitative samples

需要保存对应文件名，便于论文图表复现。

### 7. RELLIS-3D cross-dataset 结果

如果论文周期允许，非常推荐。

## 写论文时还需确认

### 8. 最终会议/模板

需要知道：

- 目标 EI 会议名称；
- 页数限制；
- IEEE / Springer / ACM 等模板；
- 投稿截止时间。

这会影响实验数量和论文结构。

### 9. 参考文献范围

建议最终检索近 3–5 年：

- lightweight segmentation
- boundary-aware semantic segmentation
- off-road segmentation/traversability
- RK3588 / edge AI deployment（如有学术文献）

---

# 19. 当前阶段总体判断

目前工作已经不再只是“BiSeNetV2 在 RUGD 上跑一个 baseline”。

已经形成较完整的实验逻辑：

`Ontology analysis`
→ `Strong 3-class baseline`
→ `Boundary Supervision`
→ `Boundary Refinement`
→ `Joint model`
→ `RK3588 deployment`

其中前三个模型 A/B/C 已经形成较清晰的可重复消融证据。

目前最值得保留的结果是：

- A Test：81.18 mIoU / 31.55 BF1-Trav
- B Test：81.22 mIoU / 34.19 BF1-Trav
- C Test：80.98 mIoU / 35.64 BF1-Trav

即：

- B 保持 mIoU，同时提高 boundary quality；
- C 进一步提高 boundary quality，但存在轻微 mIoU trade-off；
- D 有明确的联合优化动机。

如果 Experiment D 能实现较好的 mIoU-BF1 平衡，同时在 RK3588 上保持较低延迟，那么当前工作就具备较完整的 EI 会议论文实验链条。

---

## 20. 当前关键数字速查

### Validation

| Exp | mIoU | BF1-All@3 | BF1-Trav@3 |
|---|---:|---:|---:|
| A | 82.7955% | 59.0312% | 49.5498% |
| B | **83.1244%** | **60.4139%** | 52.3167% |
| C | 83.0483% | 60.3212% | **53.0774%** |

### Test

| Exp | mIoU | BF1-All@3 | BF1-Trav@3 |
|---|---:|---:|---:|
| A | 81.1822% | 51.0820% | 31.5507% |
| B | **81.2181%** | 53.1307% | 34.1914% |
| C | 80.9800% | **53.1454%** | **35.6351%** |

### Efficiency

| Model | Params | ΔParams | RTX5090 latency @576×704 |
|---|---:|---:|---:|
| A | 5.193263 M | — | 2.644 ms |
| B deployment | ≈ A | 0 inference path | ≈ A |
| C | 5.204352 M | +0.011089 M (+0.2135%) | 2.931 ms |
| D | 待测 | 预计与 C 同级 | 待测 |

---

**文档状态：A/B/C 已整理；D、RK3588、cross-dataset、定量 FLOPs 和 qualitative figures 待补。**

