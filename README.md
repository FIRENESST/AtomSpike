# AtomSpike

屏幕语义理解（CNN + Transformer）+ 脉冲策略网络，在**键鼠原子操作**粒度上输出概率并触发。本目录是可安装、可在 CPU 上跑通 T0–T6 的完整应用实现，首战环境为合成瞄准任务（以及可选的 ViZDoom）。

设计来源：仓库根目录两份 README（低成本训练链 + 混合架构）。下面在实现里做了几处**更优解**，而不是逐行翻译文档里的规划目录。

## 相对文档的优化

| 文档原方案 | 这里的更优解 | 原因 |
|---|---|---|
| 感知 5Hz / 策略 30Hz，策略只吃缓存特征 | 5Hz 推理 + **30Hz 时序残差适配器**（当前帧与帧差） | 200ms 视觉冻结对瞄准/走位过钝，残差只增加极少算力 |
| 8-token 用完整 Transformer 解码器 | 默认 **GRU 自回归**（可选 parallel / transformer_ar） | 8 步依赖够用 GRU，30Hz 延迟更稳 |
| 从零训 SNN 或全程代理梯度 | 感知用可切换的 `Act`；默认 **脉冲策略头**；T6 把 `Act` 的 spike 开关写入 buffer，**存盘后再加载仍是 SNN** | 以前替换 nn.ReLU 后 load 回 ANN，转换结果会蒸发 |
| 在线 RL 占卡采样 | **优势加权 BC + KL 锚定教师**（AW-BC / SPAG 离线精神） | 单机双卡也不该让采样绑死训练卡 |
| 逐游戏全参微调 | 冻 CNN 骨干 + 推理层 **显式 Q/K/V Linear 上的 LoRA**（存盘可还原） | 以前匹配 `in_proj` 名字，对融合 MHA 包装数为 0 |
| 必须装 ViZDoom 才能开发 | **SyntheticAimEnv** 闭环（走位 + 瞄准 + 点击） | 笔记本 CPU 即可验证数据/训练/转换/评测 |
| 单 one-hot 组合动作 | 8 slot：4 键状态机 + 量化鼠标 + 2 鼠标键，**非法转移 mask** | 并发按键不爆炸词表，press/hold/release 合法 |
| 「每 6 次 forward 更新上下文」 | **DualRateClock**：仿真时钟按 1/30s 推进；`play` 按墙钟 sleep 到 30Hz | 紧循环 eval 不是 30Hz；调度必须基于时间 |

LIF 膜电位跨 30Hz tick 保持，与「按下 / 持续 / 释放」同构；转换后的稀疏度作为能耗代理，而不是假设普通 GPU 上 SNN 一定更快。

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

动作 slot：`W A S D` 各 4 态（idle/press/hold/release）+ `dx dy` 量化 bin + `LMB RMB` 4 态。

## 目录

```
src/
├── atomspike/
│   ├── capture/       屏幕捕获、演示录制、研究环境注入（默认关闭）
│   ├── data/          HDF5 三元组、时间戳对齐、RA-BC / 覆盖度重加权
│   ├── models/        encoder / reasoner / temporal adapter / ANN+脉冲策略
│   ├── train/         BC → 离线 RL → LoRA PEFT → 蒸馏
│   ├── convert/       PMSM、SpikedAttention 免训练转换
│   ├── envs/          synthetic、可选 vizdoom
│   ├── eval/          成功率、延迟 p95、token 准确率、能耗代理
│   ├── runtime/       双频推理循环
│   └── cli.py
├── configs/           default.yaml / smoke.yaml
├── pyproject.toml
└── requirements.txt
```

## 训练链

1. **T0** `collect` — 脚本专家或人类演示 → HDF5 `(frame, game_state, action)`
2. **T1** `train-bc` — 行为克隆 warm-start
3. **T2** `--rabc` — 轨迹回报筛选 + 稀有动作 n-gram 上采样（PostBc-lite）
4. **T3** `train-rl` — 优势加权离线更新 + KL 正则
5. **T4** `train-peft` — 冻骨干 + LoRA
6. **T5** 蒸馏（`train.peft_train.distill_ann_to_small`）
7. **T6** `convert` — PMSM（默认 T=1）或 SpikedAttention

## 安装与运行

在 `src` 目录：

```bash
pip install -e .
python -m atomspike verify --config configs/smoke.yaml
python -m atomspike info --config configs/smoke.yaml
python -m atomspike smoke --workdir runs/smoke
```

`verify` 会断言三件事：T4 LoRA 包装数 > 0 且 save/load 后输出不变；T6 PMSM 改变输出且加载后仍是 spike 模式；仿真时钟 30 次策略 tick 对应 5 次感知，墙钟 pace 接近 30Hz / 5Hz。

分步：

```bash
python -m atomspike collect --out runs/demos.h5 --episodes 40
python -m atomspike train-bc --data runs/demos.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/demos.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/demos.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm
python -m atomspike eval --ckpt runs/snn.pt --episodes 12
python -m atomspike play --ckpt runs/snn.pt --episodes 2
```

合规：只在合成环境、ViZDoom / MineRL 一类研究环境里跑。`play --inject` 会被拒绝——不要接到线上对战。

## 硬件角色（与文档一致）

- 96GB 卡：骨干预训练、逐游戏 LoRA、T6 转换实验
- 16GB 卡：环境评估、自博弈采样、30Hz 延迟测试
- 不做不对等双卡 DDP；7B+ 仅在大卡上 QLoRA（本 MVP 骨干约 1M 级，对标 DOOM 小模型）

## 指标

| 维度 | 命令 | 目标 |
|---|---|---|
| 控制 | `eval` success_rate | 对标专家 / 公开基线 |
| 时序 | expert_token_acc | 与 ANN 差距 ≤ 5% |
| 实时 | `eval` scheduled_*_hz / `play` policy_hz | 仿真 30/5Hz；play 墙钟 ≈30Hz |
| 效率 | convert 的 sparsity | 低于 dense ANN 的代理能耗 |
