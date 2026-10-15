# DiT3D / Tensor-DiT 代码审计（2026-10-14）

审计对象：本地提交 `a37ac60`，以及下载的 `flow3d_full_3354345` 全量测试报告。
审计方式：官方源码对照、CPU 单元测试、独立数值探针、真实数据真值场回放。
本次没有修改训练、采样或汇总代码，没有重新训练，也没有重新提交服务器作业。

**结论：没有找到能直接证明本批 DiT3D/Tensor-DiT 效果差的核心公式或张量轴实现错误。确认存在验证与报告缺口，以及未被本批参数触发的输入校验/汇总缺陷。采样稳定性是重点嫌疑，但仍需要实际故障 checkpoint 的逐步诊断。**

## 1. 结果事实与审计边界

数据包包含 63 个配置、每个配置 101 个测试流场，共 6363 条记录。各架构为 7 种粒子数 × 3 个训练种子 × 101 个测试流场 = 2121 次案例运行。

| 架构 | 单案例相对 L2：最小 / 中位数 / 最大 | 含无效观测轨迹点的案例运行 | 含无效探针轨迹点的案例运行 |
|---|---|---:|---:|
| UNet3D | 0.129 / 0.586 / 2.839 | 0 / 2121 | 0 / 2121 |
| DiT3D | 0.852 / 1.183 / 2.963 | 0 / 2121 | 0 / 2121 |
| Tensor-DiT | 1473.49 / 8950.97 / 65096.96 | 2121 / 2121 | 2121 / 2121 |

这里“含无效轨迹点”表示至少一个粒子时刻无效，不表示所有粒子、所有时刻无效。相对 L2 在未知内部区域计算；该区域零预测的相对 L2 为 1。DiT3D 的 21 个配置分别取测试均值后全部大于 1。

源数据：`outputs/hpc_reports/flow3d_full_3354345/metrics/per_case.jsonl`、`metrics/summary.json`。

当前本地下载的正式训练权重只有 12 个 UNet3D checkpoint；本报告没有加载本批正式 DiT3D/Tensor-DiT 权重。模拟误差放大和随机初始化网络探针不能冒充故障模型的实测采样轨迹。

对训练版本 `b86b6f0b2c10` 与评估版本 `a37ac60` 的比较中，相关 DiT、Tucker、diffusion 和训练核心源码没有变化；采样脚本的变化是 checkpoint 跨平台加载和记录元数据。

## 2. 已核验，未发现错误的部分

| 检查项 | 实测证据 |
|---|---|
| DiT3D 切块/还原与三维轴顺序 | 独立逐块枚举、梯度和 round-trip 误差均为 0；线性 patch embedding 与等价 stride-p Conv3d 误差为 0 |
| DiT attention、adaLN shift/scale/gate | 与独立方程展开最大差 `2.22e-16` |
| 零初始化能否学习 | toy 5 步更新 loss 从 0.929 降至 0.732；条件、时间、attention 梯度可激活；这不代表真实任务已收敛 |
| CFG 空条件 | 训练置零条件与采样 `None` 条件等价，输出差 0 |
| 模型配置/state_dict 恢复 | 往返输出差 0，采样使用 eval 模式 |
| EMA 与 best checkpoint | 验证使用 EMA encoder + EMA denoiser，默认采样也使用相同 EMA 权重；不是 raw/EMA 错配 |
| DDIM/epsilon→x0/CFG 公式 | 独立转写官方公式后，匹配参数下 float32 最大相对差 `4.79e-6`，一致 float64 系数下 `2.13e-12` |
| DDIM oracle 回放 | 已知正确去噪器、50 步采样，float32 最大误差 `5.96e-8` |
| Tucker 非平凡标准化正反变换 | 非零 mean、非均匀 std 下重建相对误差 `2.33e-7` |
| Tucker Procrustes 对齐 | 旋转后对齐、解码相对误差 `6.54e-7`，未发现转置或因子轴错误 |
| 数据与轨迹对齐 | 6 个真实案例用真值场回放，1 子步 RMSE `3.38e-6～7.81e-6 cells`；全部轨迹有效 |

参考 [官方 DiT 模型](https://github.com/facebookresearch/DiT/blob/main/models.py)、[官方训练](https://github.com/facebookresearch/DiT/blob/main/train.py)、[OpenAI diffusion](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py)。上述 DDIM 对照是独立转写方程的探针，不是执行下载的整个上游训练系统。

## 3. 需要处理的验证与报告缺口

**A. checkpoint 选择缺少生成质量验证。**

`scripts/train_sparse_track_diffusion.py:449`、`:507`、`:926`：每个验证案例固定一个扩散 timestep/noise，按验证总损失选 best；本批配置的物理辅助项权重为默认零时，总损失就是 epsilon MSE。没有定期跑完整生成并检查流场误差、速度幅度和轨迹有效性。

101 个验证案例、seed 31/32/33 的 timestep 最大值分别为 991/995/992，没有 999；`t>=990` 分别只有 1/1/3 个。固定随机输入本身合理，但这组验证不能检验最敏感的采样起始步，更不能代替多步生成验证。这是检查盲点，尚不是已证实的发散根因。

**B. 完整性报告没有充分突出物理失效。**

`scripts/sample_sparse_track_diffusion.py:200` 的 scientific 标记检查数据划分与来源；`scripts/run_hpc_matrix.py:412` 和报告的 complete 检查执行与结果完整性。它们不表示重建质量合格。Tensor 的无效轨迹标记已写入 JSON 和案例图，但主汇总依然显示 complete、零完整性 warnings，容易被误读。

修复方向是分别报告“执行完整性、来源协议、物理有效性、精度基线”，保留所有失败重建案例与失败率；不能删掉差结果后计算更好看的均值。

## 4. 可复现的非默认参数缺陷（未导致本批异常）

**C. eta 输入与 DDIM 方差缺少一致性校验。**

`src/flow_observation/diffusion.py:227` 接受任意非负 eta，`:319` 将负的方向方差截成零。cosine 的 999→979 步中，eta=1.01 会产生约 `-0.020078` 的方向方差，仍继续采样。应拒绝不满足方差约束的 eta，或采用足够安全的 `0<=eta<=1` 范围。**本批 eta=1，没有触发。**

**D. sweep 指标汇总会静默省略非有限值。**

`scripts/run_sparse_particle_sweep.py:563` 的 `_numeric_metrics` 省略 NaN/Inf，`:606` 计算均值但没有逐指标有效样本数。最小复现中，两条 completed 记录的指标为 1.0/NaN，summary 仍显示 completed=2、complete=true、均值=1.0、无警告；矩阵验证也接受。后续可视化报告会拒绝必需指标非有限的案例并警告，但中间 summary 本身仍可能误导。**本批 6363 条记录的数值指标全部有限，没有触发。**

另有适用范围限制：轨迹编码器将时间归一化后丢失绝对时长，所有时间乘二时条件向量不变。本批所有生成案例的时长均为 120，因此它不是当前差异原因；以后混合观测时长时必须处理。

## 5. 与参考仓库不同的设计，不应直接称为 bug

本项目是针对投影粒子轨迹的自定义适配，不是官方预训练 DiT 或 DiffATS 的直接复现。

| 项目 | 本项目矩阵/默认设置 | 参考实现 |
|---|---|---|
| DiT 宽度/深度 | 192 / 4，DiT3D denoiser 约 299 万参数 | 官方最小命名 DiT-S 为 384 / 12 |
| 条件输入 | 粒子轨迹池化为单个 128 维向量 | DiT 原版用类别；DiffATS Burgers 用初始场条件 tokens 和物理参数 |
| 位置编码 | 三维坐标 MLP | DiT 原版固定二维 sin/cos |
| 时间 MLP 初始化 | Xavier | DiT 原版 normal std=0.02 |
| 训练配方 | batch 1，lr 2e-4，EMA .999 | DiT 原版默认 batch 256，lr 1e-4，EMA .9999 |
| 噪声与采样 | cosine 1000，50 步 DDIM，eta=1，CFG=1.5 | DiT 原版默认 linear；DiffATS Burgers 配置 linear 1000、sample 250，训练可视化调用 p_sample_loop |
| Tucker 标准化 | 每个潜变量坐标 train-only mean/std | DiffATS Burgers 按每类因子/核心使用整体 mean/std |

对应来源：[DiT models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)、[DiT train.py](https://github.com/facebookresearch/DiT/blob/main/train.py)、[DiT diffusion 配置](https://github.com/facebookresearch/DiT/blob/main/diffusion/__init__.py)、[DiffATS Burgers 配置](https://github.com/JinhuaLyu/DiffATS/blob/main/exps/tensor_physics/exp_burgers_2d/configs/train_v1.yaml)、[DiffATS 训练/采样](https://github.com/JinhuaLyu/DiffATS/blob/main/exps/tensor_physics/exp_burgers_2d/train/train_burgers_2d.py)、[DiffATS 标准化/重建](https://github.com/JinhuaLyu/DiffATS/blob/main/exps/tensor_physics/exp_burgers_2d/train/dataset_burgers_2d.py)。

不能因为本项目没有 learned variance 就称实现错误：当前输出 epsilon 与 DDIM 接口匹配。同理，sin/cos 排列不同不是训练/采样失配，模型规模小也不是程序错误。

**对之前判断的修正：没有采样截断、没有对生成因子再做 QR，均不能单独作为错误证据。** [官方 DiT 采样](https://github.com/facebookresearch/DiT/blob/main/sample.py) 明确关闭 clip_denoised；DiffATS 上述 Burgers 分支也关闭截断，直接反标准化后多线性重建。对因子单独做 QR 而不相应变换核心会改变流场，不能这样“修复”。

但当前组合存在可量化的数值敏感性：cosine 1000 的末步噪声预测误差到 x0 的放大约 20291 倍，linear 1000 约 157 倍。这一性质也存在于官方 cosine 公式，不是本项目算错。合成探针中，Tucker 核心与三个因子分别乘二，解码流场幅度乘 16；这仅证明多线性放大机制，不证明真实生成因子恰好这样失稳。

不能把已有 cosine checkpoint 直接换成 linear 日程称作修复；这会改变训练与采样的对应关系。新日程应重新训练，或采用有明确推导且保持原训练时间/SNR 映射的采样方案。

## 6. 后续最小诊断顺序

1. 固定一个现有 DiT3D 和一个 Tensor-DiT checkpoint，读取 model_config、训练历史、codec 统计量；先验证模型/EMA/normalization 身份。
2. 在固定少量**验证集**案例上，记录各采样步 epsilon、x0、latent 范数，Tensor 同时记录核心/因子范数、正交残差与解码速度；定位第一个异常步骤。
3. 保持 checkpoint 和初始噪声一致，对照正确条件、打乱条件、零条件；判断轨迹条件是否发挥作用。
4. 在验证集做受控的 CFG/eta/采样步数对照，每次只改一个因素；先分清采样失稳与模型未收敛，再考虑改变训练配方。
5. 新训练补充定期生成验证与物理失效报警。方案固定后再重新评估测试集，避免继续根据测试集挑最优设置。

## 7. 验证产物

CPU 基础测试：

```powershell
python -m pytest tests/test_tensor_space.py tests/test_dit.py tests/test_diffusion.py tests/test_dit_cli_integration.py -q --basetemp=outputs/code_audit_20261014/pytest_core
```

结果：**42 passed in 28.95s**。通过单元测试不代表真实任务收敛或 GPU 上所有数值路径都已验证。

独立探针和输出位于：

- `outputs/code_audit_20261014/dit_reference/probe.py` / `probe_results.json`
- `outputs/code_audit_20261014/sampler/probe_sampler.py` / `sampler_probe_results.json`
- `outputs/code_audit_20261014/training/probe_training_pipeline.py` / `probe_results.json`
- `outputs/code_audit_20261014/training/probe_nan_omission.py` / `nan_omission_results.json`
- `outputs/code_audit_20261014/probe_tensor_codec.py` / `tensor_codec_probe_results.json`

本次下载的参考源码保存在 `outputs/code_audit_20261014/reference/`。正式模型源代码未变更；以上审计应与之后的修复提交分别记录。
