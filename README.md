# AtomSpike

面向游戏的原子级键鼠操作智能体：CNN + Transformer 感知、脉冲神经网络策略头，在 **W/A/S/D/鼠标移动/鼠标点击** 的原子粒度上输出概率并触发。为单卡低算力环境设计——笔记本 CPU 即可跑通全链路，单张 16GB 消费级 GPU 即可完成全部训练与推理。

## 核心特性

* **双频推理架构**：感知层 5Hz 重推理 + 30Hz 帧差残差适配器，视觉理解与操作响应互不拖累

* **脉冲策略头**：LIF 膜电位跨 30Hz tick 保持，与键鼠「按下/持续/释放」天然同构；支持免训练 ANN→SNN 转换（PMSM / SpikedAttention）

* **原子动作空间**：8 slot（4 键状态机 + 量化鼠标 + 2 鼠标键），非法转移被 mask，并发按键不爆炸词表

* **离线优先训练链**：BC → RA-BC 数据筛选 → 优势加权离线 RL → LoRA PEFT → 蒸馏，全程不占卡采样

* **低算力训练基建**：5 状态原子检查点、WSD 学习率调度、Muon+AdamW 混合优化器、bf16 混合精度

* **零依赖开发环境**：SyntheticAimEnv 合成瞄准环境闭环验证，无需安装 ViZDoom 即可开发

## 架构

```
帧 30–60fps ──► CNN Encoder ──► Transformer Reasoner (5Hz, DualRateClock)
                      │                    │
                      └── 帧差残差(30Hz) ──┴──► Spike Policy (默认) / ANN ──► 8-token
                                                                    │
                                                              温度采样 / argmax
                                                                    │
                                                              合成环境 或 研究环境
```

* **感知（5Hz）**：CNN 编码帧 → Transformer 推理层融合游戏状态 token

* **策略（30Hz）**：时序残差适配器补充帧差信息 → GRU 自回归解码 8 个动作 token

* **动作空间**：`W A S D` 各 4 态（idle/press/hold/release）+ `dx dy` 量化 bin + `LMB RMB` 4 态

关键实现细节：

* 激活统一走 `models/activations.py::Act`，spike 开关以 buffer 存盘——转换后的 SNN 存盘再加载仍是 SNN

* 自注意力是 `models/attention.py::MultiHeadSelfAttention`，显式拆分 Q/K/V Linear——LoRA 有真实的包装目标、T6 可对注意力打阈值

## 训练链

| 阶段 | 命令                     | 内容                                             |
| -- | ---------------------- | ---------------------------------------------- |
| T0 | `collect`              | 脚本专家或人类演示 → HDF5 `(frame, game_state, action)` |
| T1 | `train-bc`             | 行为克隆 warm-start                                |
| T2 | `train-bc --rabc`      | 轨迹回报筛选 + 稀有动作 n-gram 上采样                       |
| T3 | `train-rl`             | 优势加权离线更新 + KL 锚定教师                             |
| T4 | `train-peft`           | 冻结骨干 + LoRA 逐游戏适配                              |
| T5 | `distill_ann_to_small` | 大模型蒸馏到小模型                                      |
| T6 | `convert`              | PMSM（T=1）或 SpikedAttention 免训练转换               |

## 安装与运行

要求 Python ≥ 3.10。在 `src` 目录：

```bash
pip install -e .

# 快速验证（CPU 可跑）
python -m atomspike verify --config configs/smoke.yaml
python -m atomspike info --config configs/smoke.yaml
python -m atomspike smoke --workdir runs/smoke
```

`verify` 跑 4 组回归：LoRA 包装数 > 0 且 save/load 后输出不变；PMSM 转换改变输出且存盘再加载仍是 spike 模式；仿真时钟 30 次策略 tick 对应 5 次感知；realtime pace 墙钟接近 30Hz / 5Hz。

完整流水线：

```bash
python -m atomspike collect --out runs/demos.h5 --episodes 40
python -m atomspike train-bc --data runs/demos.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/demos.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/demos.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm
python -m atomspike eval --ckpt runs/snn.pt --episodes 12
python -m atomspike play --ckpt runs/snn.pt --episodes 2
```

## 目录结构

```
src/
├── atomspike/
│   ├── capture/       屏幕捕获、演示录制、研究环境注入（默认关闭）
│   ├── data/          HDF5 三元组、时间戳对齐、RA-BC / 覆盖度重加权
│   ├── models/        encoder / reasoner / attention / activations / temporal / 策略头
│   ├── train/         BC / 离线 RL / PEFT / 蒸馏 + 训练基建
│   │   ├── common.py  5 状态原子检查点
│   │   ├── optim.py   Muon + AdamW 混合优化器
│   │   └── sched.py   WSD 学习率调度
│   ├── convert/       PMSM、SpikedAttention 免训练转换
│   ├── envs/          synthetic、可选 vizdoom
│   ├── eval/          成功率、延迟 p95、token 准确率、能耗代理
│   ├── runtime/       双频时钟 + 推理循环
│   └── cli.py
├── verify.py          回归：LoRA 往返 / PMSM 存盘持久 / 双频时钟
├── configs/           default.yaml / smoke.yaml
├── pyproject.toml
└── requirements.txt
```

## 训练基建（低算力高效率）

所有机制不限定具体显卡型号——从笔记本 CPU 到单张 16GB 消费卡均可复用：

* **5 状态原子检查点**：模型 + 优化器 + LR 调度 + RNG + DataLoader。先写 `.tmp` 再 `os.replace` + SHA-256 校验，断电不会出现半个文件；keep-k 滚动清理

* **WSD 三阶段 LR**：5% warmup / 85% stable / 10% decay。稳定期任意存档都能补一段 decay 出货，训练随时可中断、中断不报废

* **Muon + AdamW 混合**：2D 权重矩阵走 Muon（Newton-Schulz 正交化动量），bias/LayerNorm/标量走 AdamW

* **bf16 混合精度**：默认开启，CUDA/CPU 都通过 `torch.autocast`；可选 `torch.compile`

* **梯度累积**：`grad_accum > 1` 时 micro-batch 累积后再 `optimizer.step()`，数学等价大 batch

通过 `configs/*.yaml` 的 `train.*` 配置：

```yaml
train:
  precision: bf16         # fp32 / fp16 / bf16
  compile: false          # torch.compile
  optimizer: adamw        # adamw / muon / muon_adamw
  lr_schedule: wsd        # constant / wsd
  wsd_warmup_frac: 0.05
  wsd_decay_frac: 0.10
  grad_accum: 1
  ckpt_keep_k: 3
```

## 指标

| 维度 | 命令                                           | 目标                      |
| -- | -------------------------------------------- | ----------------------- |
| 控制 | `eval` success\_rate                         | 对标专家 / 公开基线             |
| 时序 | expert\_token\_acc                           | SNN 与 ANN 差距 ≤ 5%       |
| 实时 | `eval` scheduled\_\*\_hz / `play` policy\_hz | 仿真 30/5Hz；play 墙钟 ≈30Hz |
| 效率 | `convert` 的 sparsity                         | 低于 dense ANN 的代理能耗      |

## 合规

只在合成环境、ViZDoom / MineRL 一类研究环境里运行。`play --inject` 会被拒绝——请勿接入线上对战。
