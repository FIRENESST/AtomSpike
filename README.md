# AtomSpike

屏幕语义理解（CNN + Transformer）+ 脉冲策略网络，在**键鼠原子操作**粒度上输出概率并触发。首选目标是 **Minecraft**（单机 / LAN 研究采集）；合成瞄准环境只用于算法冒烟。

设计来源：仓库根目录两份 README（低成本训练链 + 混合架构）。下面在实现里做了几处**更优解**，而不是逐行翻译文档里的规划目录。

## 相对文档的优化

| 文档原方案                       | 这里的更优解                                                                     | 原因                                |
| --------------------------- | -------------------------------------------------------------------------- | --------------------------------- |
| 感知 5Hz / 策略 30Hz，策略只吃缓存特征   | 5Hz 推理 + **30Hz 时序残差适配器**（当前帧与帧差）                                          | 200ms 视觉冻结对瞄准/走位过钝，残差只增加极少算力      |
| 8-token 用完整 Transformer 解码器 | 默认 **GRU 自回归**（可选 parallel / transformer\_ar）                              | 8 步依赖够用 GRU，30Hz 延迟更稳             |
| 从零训 SNN 或全程代理梯度             | 感知用可切换的 `Act`；默认 **脉冲策略头**；T6 把 `Act` 的 spike 开关写入 buffer，**存盘后再加载仍是 SNN** | 以前替换 nn.ReLU 后 load 回 ANN，转换结果会蒸发 |
| 在线 RL 占卡采样                  | **优势加权 BC + KL 锚定教师**（AW-BC / SPAG 离线精神）                                   | 单机双卡也不该让采样绑死训练卡                   |
| 逐游戏全参微调                     | 冻 CNN 骨干 + 推理层 **显式 Q/K/V Linear 上的 LoRA**（存盘可还原）                          | 以前匹配 `in_proj` 名字，对融合 MHA 包装数为 0  |
| 必须装 ViZDoom 才能开发            | **SyntheticAimEnv** 闭环（走位 + 瞄准 + 点击）                                       | 笔记本 CPU 即可验证数据/训练/转换/评测           |
| 单 one-hot 组合动作              | 8 slot：4 键状态机 + 量化鼠标 + 2 鼠标键，**非法转移 mask**                                 | 并发按键不爆炸词表，press/hold/release 合法   |
| 「每 6 次 forward 更新上下文」       | **DualRateClock**：仿真时钟按 1/30s 推进；`play` 按墙钟 sleep 到 30Hz                   | 紧循环 eval 不是 30Hz；调度必须基于时间         |
| 单一固定学习率                     | **WSD 三阶段调度**（5% warmup / 85% stable / 10% decay），可任意中断出货                  | 稳定期任意存档+补一段 decay 即可用，训练不怕断电      |
| 默认 fp32                     | **bf16 混合精度**（CUDA + CPU autocast），可选 `torch.compile`                      | 16GB 单卡装下更大 batch；吞吐 2-3 倍        |
| 普通 AdamW                    | **Muon + AdamW 混合**（矩阵参数走 Muon，标量/LayerNorm 走 AdamW）                       | 矩阵优化快 1.3-1.4 倍；优化器状态省 75%        |
| 只存模型权重                      | **5 状态原子检查点**：模型+优化器+LR 调度+RNG+DataLoader，先写 .tmp 再 rename，keep-k 滚动       | 断电/中断随时续训，不会出现半个文件                |

LIF 膜电位跨 30Hz tick 保持，与「按下 / 持续 / 释放」同构；转换后的稀疏度作为能耗代理，而不是假设普通 GPU 上 SNN 一定更快。

## 架构

```
帧 30fps 采集 ──► CNN Encoder ──► Transformer Reasoner (5Hz, DualRateClock)
                      │                    │
                      └── 帧差残差(30Hz) ──┴──► Spike Policy (默认) / ANN ──► 键鼠 slots
                                                                    │
                                                              温度采样 / argmax
                                                                    │
                                                              Minecraft 采集 或 合成冒烟
```

动作 slot（Minecraft 默认 12）：`W A S D SPACE SHIFT CTRL E` 各 4 态 + 量化鼠标 `dx dy` + `LMB RMB`。合成冒烟仍是 4 键 / 8 slot。

两个可交换的实现点：激活统一走 `models/activations.py::Act`，spike 开关以 buffer 存盘，T6 转换后加载不蒸发；自注意力是 `models/attention.py::MultiHeadSelfAttention`，显式拆分 Q/K/V Linear，使 T4 LoRA 有真实的包装目标、T6 可对注意力打阈值。

## 目录

```
src/
├── atomspike/
│   ├── capture/       Minecraft 30fps 采集（窗口/键鼠/HUD/NBT/sidecar）
│   ├── envs/          synthetic 冒烟、minecraft 人机采集入口、可选 vizdoom
│   ├── data/          HDF5 三元组 + StreamingDemoWriter（30fps 追加）
│   ├── models/        encoder / reasoner / attention / activations / temporal / 策略头
│   ├── train/         BC → 离线 RL → LoRA PEFT → 蒸馏
│   ├── convert/       PMSM、SpikedAttention
│   ├── eval/          成功率、延迟 p95、token 准确率
│   ├── runtime/       DualRateClock + 推理循环
│   ├── verify.py      LoRA / PMSM / 双频 / 采集器 dry-run
│   └── cli.py
├── configs/           default.yaml（Minecraft）/ minecraft.yaml / smoke.yaml（合成）
├── pyproject.toml
└── requirements.txt
```

## 训练链

1. **T0** `collect-mc` — Minecraft 人类演示 30fps → HDF5 `(frame, game_state, action)`
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

`verify` 跑 5 组回归：LoRA / PMSM 存盘 / 仿真 30·5Hz / 墙钟 pace / Minecraft 采集器 dry-run。

Minecraft 30fps 采集（窗口画面 + 生命/饥饿/坐标等重点数据 + 键鼠）：

```bash
pip install -e ".[capture]"
python -m atomspike collect-mc --list-windows
python -m atomspike collect-mc --out runs/mc.h5 --seconds 120
```

开着单机或局域网世界。`F9` 结束，`F8` 打 episode 标记。每帧写入 HDF5：`frames`（默认 90×160 RGB）、19 维 `game_state`、12 维动作 token、`keys_held`、`mouse_xy`（视角 delta）、`cursor_xy`（窗口内 0–1 光标）、`wheel`、`facts`、`timestamps_ns`。

物品栏 / 箱子 / 熔炉等界面打开时，鼠标槽位改为**窗口内绝对光标**（不是视角 delta），并记下左/右键、滚轮、SHIFT+点击。sidecar 还会带上 `hovered_slot`（鼠标下的槽）。走路视角仍用 raw mouse delta。旧的 16 维 `runs/mc.h5` 需要重新采集才能和当前配置一起训练。

状态融合（后者覆盖前者）：HUD 像素估计 ← 存档 NBT（约 2Hz，跟自动保存）← **sidecar**（本机 UDP `127.0.0.1:21337` 或 JSON 文件，实时）。

sidecar 不是注入、也不是读内存：游戏里另开一个客户端模组，把**本地玩家自己的**生命/坐标/朝向打成 JSON 发给采集器。没有 sidecar 时画面和键鼠仍是 30fps，生命和 xyz 只能靠 HUD（不准）或 NBT（准但滞后）。实时日志里 `src=sidecar` 表示这一路生效。

先不装模组、只验证采集器能否吃 UDP：

```bash
python -m atomspike sidecar-send --count 40 --hz 20
```

（另开一个终端跑 `collect-mc`，应看到 `src=sidecar` 以及假的 hp/xyz。）

真游戏实时数据：`tools/mc-sidecar` 的 jar 放到该版本 `mods`（需 Fabric API）。**完全退出并重启**客户端后进单机/LAN，采集日志应为 `src=sidecar`，生命/坐标/朝向跟你走。开物品栏时还会有 `cursor=`、`hover=`、`gui`。协议示例：

```json
{"hp": 20, "hunger": 18, "x": 12.5, "y": 64.0, "z": -8.1, "yaw": 90.0, "pitch": 12.0, "hotbar_slot": 0, "gui_open": true, "mouse_nx": 0.42, "mouse_ny": 0.61, "hovered_slot": 10}
```

不要对公众服做自动化。无窗口自检：`python -m atomspike collect-mc --dry-run --seconds 1 --out runs/mc_dry.h5`。正式采集默认弹出 **采集画面** 预览窗（写入 HDF5 的 854×480 帧）；`--no-preview` 关掉。

采集之后的训练与预测（配置用 `default.yaml` / `minecraft.yaml`，不要用 smoke）：

```bash
python -m atomspike train-bc --data runs/mc.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/mc.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/mc.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm
python -m atomspike eval --ckpt runs/snn.pt --data runs/mc.h5
python -m atomspike predict-mc --ckpt runs/snn.pt --seconds 30 --dump runs/pred.jsonl
```

`predict-mc` 以 30fps 抓窗口 + 状态，输出键鼠 token，**不注入操作系统**。`play --inject` 仍然拒绝。

分步（合成冒烟仍可用 `collect --config configs/smoke.yaml`）：

```bash
python -m atomspike collect --config configs/smoke.yaml --out runs/demos.h5 --episodes 40
python -m atomspike train-bc --data runs/demos.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/demos.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/demos.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm
python -m atomspike eval --ckpt runs/snn.pt --data runs/mc.h5
python -m atomspike predict-mc --ckpt runs/snn.pt --seconds 30 --dump runs/pred.jsonl
```

合规：Minecraft 只做单机 / LAN 研究采集。`play --inject` 拒绝接到线上对战。

## 硬件角色（与文档一致）

* 96GB 卡：骨干预训练、逐游戏 LoRA、T6 转换实验

* 16GB 卡：环境评估、自博弈采样、30Hz 延迟测试

* 不做不对等双卡 DDP；7B+ 仅在大卡上 QLoRA（本 MVP 骨干约 1M 级，对标 DOOM 小模型）

## 训练基建（低算力高效率）

本实现按「单卡 16GB 闭环」设计训练基建，但所有机制不限定具体显卡型号——任何同级消费卡都能复用：

* **5 状态原子检查点**：模型 + 优化器 + LR 调度 + RNG + DataLoader。先写 `.tmp` 再 `os.replace` + SHA-256 校验，断电不会出现半个文件；keep-k 滚动（默认 3）+ best 额外存。

* **WSD 三阶段 LR**：5% warmup / 85% stable / 10% decay。稳定期任意存档都能补一段 decay 出货，训练随时可中断。

* **Muon + AdamW 混合**：2D 权重矩阵走 Muon（Newton-Schulz 正交化动量），bias/LayerNorm/标量走 AdamW。

* **bf16 混合精度**：默认开启，CUDA/CPU 都通过 `torch.autocast`；可选 `torch.compile`（默认关，配置开）。

* **梯度累积**：`grad_accum > 1` 时 micro-batch 累积后再 `optimizer.step()`，数学等价大 batch。

在 `configs/*.yaml` 里通过 `train.*` 配置：

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
| 时序 | expert\_token\_acc                           | 与 ANN 差距 ≤ 5%           |
| 实时 | `eval` scheduled\_\*\_hz / `play` policy\_hz | 仿真 30/5Hz；play 墙钟 ≈30Hz |
| 效率 | convert 的 sparsity                           | 低于 dense ANN 的代理能耗      |

