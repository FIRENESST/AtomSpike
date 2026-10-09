# AtomSpike

屏幕视觉表征（CNN + Transformer）+ 脉冲策略网络，在**键鼠原子操作**粒度上输出概率。真正的屏幕语义尚在验证：新增的十字准星概念探针需要带标签的新录制和独立留出评估，不能把原有未监督的网格类别当成理解。首选目标是 **Minecraft 单机研究采集**；合成瞄准环境只用于算法冒烟。OS 键鼠注入只会在带本次随机 nonce 回显的新鲜 sidecar 明确证明“集成式单机服务器”、真实游戏窗口和前台焦点同时成立时启用。

Git 仓库在 `src/`。日常命令见上一级 `Reference Documentation/README-操作命令.md`；早期设计稿是同目录的 `README.md`。Sidecar 模组在 `tools/mc-sidecar`。下面是实现相对那份设计稿的差异，不是逐行翻译。

## 相对文档的优化

| 文档原方案                       | 这里的更优解                                                                     | 原因                                |
| --------------------------- | -------------------------------------------------------------------------- | --------------------------------- |
| 感知 5Hz / 策略 30Hz，策略只吃缓存特征   | Minecraft **10Hz 推理** + **30Hz 时序残差**（当前帧与帧差）；冒烟仍 5Hz                     | 200ms 视觉冻结对瞄准/走位过钝，残差只增加极少算力      |
| 8-token 用完整 Transformer 解码器 | 默认 **GRU 自回归**（可选 parallel / transformer\_ar）                              | slot 逐步依赖够用 GRU，30Hz 延迟更稳          |
| 从零训 SNN 或全程代理梯度             | 感知用可切换的 `Act`；T6 是**逐层校准的、有损二值激活实验**，会保存并报告实际 logit 漂移 | 不能把单步二值化误称为无损 ANN→SNN 或能耗实测 |
| 在线 RL 占卡采样                  | 离线 HDF5 用**优势加权 BC + KL**；单机闭环可选**采样策略梯度 + value baseline + entropy + 冻结教师 KL** | 缺少离线回报时命令拒绝；在线不再把 argmax 当监督标签 |
| 逐游戏全参微调                     | 冻 CNN 骨干 + 推理层 **显式 Q/K/V Linear 上的 LoRA**（存盘可还原）                          | 以前匹配 `in_proj` 名字，对融合 MHA 包装数为 0  |
| 必须装 ViZDoom 才能开发            | **SyntheticAimEnv** 闭环（走位 + 瞄准 + 点击）                                       | 笔记本 CPU 即可验证数据/训练/转换/评测           |
| 单 one-hot 组合动作              | Minecraft **13 slot**（8 键 + 量化鼠标 + 滚轮）；冒烟仍 8 slot，**非法转移 mask**           | 并发按键不爆炸词表，press/hold/release 合法   |
| 「每 6 次 forward 更新上下文」       | **DualRateClock**：仿真时钟按 1/30s 推进；`play` 按墙钟 sleep 到 30Hz                   | 紧循环 eval 不是 30Hz；调度必须基于时间         |
| 单一固定学习率                     | **WSD 三阶段调度**（5% warmup / 85% stable / 10% decay），可任意中断出货                  | 稳定期任意存档+补一段 decay 即可用，训练不怕断电      |
| 默认 fp32                     | **bf16 混合精度**（CUDA + CPU autocast），可选 `torch.compile`                      | 16GB 单卡装下更大 batch；吞吐 2-3 倍        |
| 普通 AdamW                    | **Muon + AdamW 混合**（矩阵参数走 Muon，标量/LayerNorm 走 AdamW）                       | 矩阵优化快 1.3-1.4 倍；优化器状态省 75%        |
| 只存模型权重                      | 原子 checkpoint：模型、优化器/调度器/GradScaler、安全 RNG 与 SHA-256；加载会校验 hash | BC 分离最佳 `bc.pt` 与最新 `bc.last.pt`，固定输出名保留真实 keep-k 代际 |
| 加深就堆 30Hz 层                  | **10Hz CNN 加残差（零初始化）+ 推理 3 层 DropPath；30Hz 脉冲头仍 2 层，模拟残差从 0 学起**         | 死记硬背多半是浅视觉；预测仍 1 步 LIF，不增加 spike 展开 |

默认策略训练和推理都使用 1 步 LIF。逐层二值转换使用完整演示输入校准；有多段录制时在录制级留出集报告发放率和保真度，只有一段时会明确标为复用校准集。它们不是功耗测量，也不代表普通 GPU 会更快。

## 架构

```
帧 30fps 采集 ──► CNN Encoder ──► Transformer Reasoner (10Hz)
                      │                    ├── SemanticPlanner（任务线）
                      │                    ├── MinecraftIterator（4×6 空间 + 成就 + GRU；GUI 时冻结世界格子）
                      │                    ├── SpikeCurlScene（空间 LIF 旋度场景指纹）
                      │                    ├── SpikeGroupCoalition（发放集团 + 10Hz 脉冲自注意；活跃集团可在决策层下出生细分神经元）
                      │                    ├── SECA 屏语义符合注意（屏幕发放/ON 事件 top-K + 脉冲重合自注意，无 softmax）
                      │                    ├── GlobalEventSequence（全局事件槽 + 可学习时间排序）
                      │                    └── InventorySense（物品 ID + 2 步 LIF + GUI 指针）
                      └── 帧差残差(30Hz) ──┴──► Spike Policy / ANN ──► 13 槽键鼠
                                                                    │
                                                              温度采样 / argmax
                                                                    │
                                                              collect-mc 或 predict-mc（默认冻结；可选人类演示适配）
```

动作 slot（Minecraft 默认 13）：`W A S D SPACE SHIFT CTRL E` 各 4 态 + 量化鼠标 `dx dy` + `LMB RMB` + **滚轮 `WHL`**。合成冒烟仍是 4 键 / 8 slot。

两个可交换的实现点：激活统一走 `models/activations.py::Act`，spike 开关以 buffer 存盘，T6 转换后加载不蒸发；屏幕语义的脉冲自注意是 `models/sea.py` 的 **SECA**（top-K 事件 + 脉冲符合，不是对全图做 softmax）；推理层仍是 `models/attention.py::MultiHeadSelfAttention`，显式拆分 Q/K/V Linear，使 T4 LoRA 有真实的包装目标。SpikedAttention 的转换路径因尚未完成校准和保真度评测而禁用。

## 目录

```
src/
├── atomspike/
│   ├── capture/       窗口/键鼠/sidecar、collect-mc、predict-mc、inject
│   ├── envs/          synthetic 冒烟、minecraft 入口（拒绝脚本环境）、可选 vizdoom
│   ├── data/          HDF5（含 task_id/task_aux/inv_ids）+ 训练增强 + StreamingDemoWriter
│   ├── models/        encoder / reasoner / iterator / curl_scene / group_attn / sea / event_seq / inventory / planner / grow / 策略头
│   ├── train/         BC → 离线 RL → LoRA PEFT → 蒸馏
│   ├── convert/       校准 PMSM（SpikedAttention 转换暂禁用）
│   ├── eval/          成功率、延迟 p95、token 准确率
│   ├── runtime/       DualRateClock + 推理循环 + 在线自学
│   ├── viz/           采集预览、脉冲窗、MC 窗口识别叠加、注入时 overlay
│   ├── verify.py      LoRA / PMSM / 双频 / 采集 / 自学 / 规划器 / 加深恒等加载
│   └── cli.py
├── configs/           default.yaml、minecraft.yaml（MC）；smoke.yaml（合成，勿训 MC）
├── pyproject.toml
└── requirements.txt
```

## 训练链

1. **T0** `collect-mc` — Minecraft 人类演示 30fps → HDF5 `(frame, game_state, action, task_id, task_aux, inv_ids)`
2. **T1** `train-bc` — 行为克隆 warm-start
3. **T2** `--rabc` — 轨迹回报筛选 + 稀有动作 n-gram 上采样（PostBc-lite）
4. **T3** `train-rl` — 优势加权离线更新 + KL 正则
5. **T4** `train-peft` — 冻骨干 + LoRA
6. **T5** 蒸馏（`train.peft_train.distill_ann_to_small`）
7. **T6** `convert` — 校准后的 PMSM 二值激活实验（仅 T=1；必须接受有损并检查保真度）。SpikedAttention 转换目前禁用，直到具备同等校准与评测。

## 安装与运行

在 `src` 目录：

```bash
pip install -e .
python -m atomspike doctor
python -m atomspike verify --config configs/smoke.yaml
python -m atomspike info --config configs/smoke.yaml
python -m atomspike smoke --workdir runs/smoke
```

`doctor` 不加载模型，先检查 Python、必需/可选依赖和三个配置档案；环境不完整时会明确列出缺项。`verify` 会跑一组回归：LoRA / PMSM 存盘 / 仿真双频 / 墙钟 pace / 滚轮与自学 / 规划器 / 采集 dry-run / 训练预测冒烟 / 加深网络恒等加载（`depth_identity`）等。时钟数学测试仍用 5Hz；Minecraft 配置是 10Hz。`verify` 默认使用 `smoke.yaml`，其他命令默认使用生产配置 `default.yaml`。

配置加载是严格的：拼错的字段、非法端口、非本机 sidecar 地址、偶数鼠标分箱和不合理的双频设置会在启动模型前直接报出具体字段。三个内置配置也随 wheel 一起安装，因此从任意工作目录运行都能找到默认配置。

只跑快速工程测试（不执行完整模型回归）：

```bash
pip install -e ".[dev]"
pytest -q
```

Minecraft 30fps 采集（窗口画面 + 生命/饥饿/坐标等重点数据 + 键鼠）：

```bash
pip install -e ".[capture]"
python -m atomspike collect-mc --list-windows
python -m atomspike collect-mc --config configs/minecraft.yaml --out runs/mc.h5 --seconds 120
```

开着单机或局域网世界。`F9` 结束，`F8` 只打 episode 标记。每次启动采集器都会获得独立 `recording_id`，用于 train/val 隔离；它不会因同一次录制里的 F8 而改变。每帧写入 HDF5：`frames`（默认 854×480 RGB）、19 维 `game_state`、13 维动作 token（含滚轮）、`task_id` / `task_aux`（sidecar 的下一成就）、`inv_ids` / `hand_id` / `hover_id` / `carry_id`（物品栏 ID）、`keys_held`、`mouse_xy`（视角 delta）、`cursor_xy`（窗口内 0–1 光标）、`wheel`、`facts`、`timestamps_ns`。

流式 HDF5 额外带每行 `committed` 标记：异常中断后未提交的尾行会被读取/追加路径丢弃，不能再伪装成全零样本。训练会检查动作键名、19 维状态字段、长度与有限值；不匹配的旧文件必须重采，不能靠补零猜标签。原始视频超过 `train.max_resident_frames_mib` 时按需从 HDF5 取帧，避免在训练启动时把整段 480p/1080p 视频装进 RAM。

0.1.7+ sidecar 还会给新录制写入 `look_concept`：十字准星射线命中的粗视觉类别。`-1` 表示没有新鲜可靠的标签；旧录制追加时历史行仍为 `-1`，不会被当成“空气”。`train-bc` 只在有效行上训练屏幕中心探针，完整留出录制报告 `look_acc`、`macro_f1`、多数类基线与 `n_look`。新增的 13,764 参数效果探针（Minecraft 配置，暂为 ANN）仅用屏幕编码特征、概念预测和动作，预测同一 episode 内约 1.5 秒后的目标进度、所选物品增加、受伤、方块变化；sidecar 状态只生成结果标签，不作为探针输入。另有一个同容量、不读动作的对照头；它的损失不参与最佳检查点选择。训练前逐事件检查有效正负样本，完整留出录制的验证报告两种探针与训练集事件频率基线的 Brier 分数，以及各事件正负样本数。如果动作版不能胜过只看屏幕版，就不能声称模型真的利用了动作；即使胜过，也**不是**因果证明。旧录制没有新鲜 `look_concept`，效果损失会关闭；单次录制没有独立验证集时也不会制造验证分数。这仍只是**观察轨迹上的条件预测**，既未接入动作选择，也不能证明开放世界语义或因果“为什么”；实验门槛见 [SCB 研究方案](SEMANTIC_CAUSAL_SPIKES.md)。

可先用只读的干预实验工具准备下一阶段数据：`python -m atomspike plan-interventions --snapshots snapshots/a.zip snapshots/b.zip --treatment attack --control idle --goal collect_log --horizon-ticks 45 --out runs/intervention-plan.jsonl`。每个不同的世界快照只生成一对试验，两个动作的执行顺序随机；计划文件已存在时拒绝覆盖。必须在本地单人世界**分别恢复同一快照**再执行两臂，并独立记录起始状态指纹及四项结果。`python -m atomspike eval-interventions --plan runs/intervention-plan.jsonl --results runs/intervention-results.jsonl` 会拒绝缺臂、错序、重复快照、不同起始指纹、动作/时间窗不符或不完整计划，报告配对结果差和探索性精确检验。它不会运行游戏、注入输入、验证人工重置是否真实发生，也不会据此宣布因果性；结果 JSONL 格式和审计限制见 [SCB 研究方案](SEMANTIC_CAUSAL_SPIKES.md)。

物品栏 / 箱子 / 熔炉等界面打开时，鼠标槽位改为**窗口内绝对光标**（不是视角 delta），并记下左/右键、滚轮、SHIFT+点击。sidecar 0.1.5 还会带上 `hovered_slot`、`hand_item` / `hover_item` / `carry_item`，以及玩家 36 格 `inv` CRC32 和容器 `menu`。`InventorySense` 是一个零初始化的 2 步 LIF：感知 10Hz 更新物品偏置，GUI 打开时 30Hz 再跑一次（不算 CNN）。光标对格子做软指针（`hovered_slot` 上有一份指针 CE），SECA 会把光标所在屏幕格抬成事件，全局序列里也有一个指针槽。19 维 `game_state` 不变。走路视角仍用 raw mouse delta。旧的 16 维 `runs/mc.h5` 需要重新采集才能和当前配置一起训练。

状态融合（后者覆盖前者）：HUD 像素估计 ← 存档 NBT（约 2Hz，跟自动保存）← **sidecar**（本机 UDP `127.0.0.1:21337` 或 JSON 文件，实时）。

sidecar 不是注入、也不是读内存：游戏里另开一个客户端模组，把**本地玩家自己的**生命/坐标/朝向打成 JSON 发给采集器。没有 sidecar 时画面和键鼠仍是 30fps，生命和 xyz 只能靠 HUD（不准）或 NBT（准但滞后）。实时日志里 `src=sidecar` 表示这一路生效。

先不装模组、只验证采集器能否吃 UDP：

```bash
python -m atomspike sidecar-send --count 40 --hz 20
```

（另开一个终端跑 `collect-mc`，应看到 `src=sidecar` 以及假的 hp/xyz。）

真游戏实时数据：`tools/mc-sidecar` 的 jar 放到该版本 `mods`（需 Fabric API）。**完全退出并重启**客户端后进单机/LAN，采集日志应为 `src=sidecar`，生命/坐标/朝向跟你走。开物品栏时还会有 `cursor=`、`hover=`、`gui`。协议示例：

```json
{"hp": 20, "hunger": 18, "x": 12.5, "y": 64.0, "z": -8.1, "yaw": 90.0, "pitch": 12.0, "hotbar_slot": 0, "gui_open": true, "mouse_nx": 0.42, "mouse_ny": 0.61, "hovered_slot": 10, "hand_item": "stone_pickaxe", "hover_item": "cobblestone"}
```

不要对公众服做自动化。无窗口自检：`python -m atomspike collect-mc --dry-run --seconds 1 --out runs/mc_dry.h5`。正式采集默认弹出 **采集画面** 预览窗（写入 HDF5 的 854×480 帧）；`--no-preview` 关掉。

采集之后的训练与预测（配置用 `default.yaml` / `minecraft.yaml`，不要用 smoke）：

```bash
python -m atomspike train-bc --data runs/mc.h5 --out runs/bc.pt
# 仅在回报完整性检查通过后：python -m atomspike train-bc --data runs/mc.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/mc.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/mc.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm --data runs/mc.h5 --allow-lossy-snn
python -m atomspike eval --ckpt runs/snn.pt --data runs/mc.h5
python -m atomspike predict-mc --config configs/minecraft.yaml --ckpt runs/snn.pt --seconds 30 --dump runs/pred.jsonl
# 只在本地单机、sidecar 护栏有效时启用模型自玩自学：
python -m atomspike predict-mc --config configs/minecraft.yaml --ckpt runs/snn.pt --inject --learn --seconds 300
```

只有录制中确实出现 advancement 进度或掉血，且有明确 episode 边界时，才应加 `--rabc` 或运行 `train-rl`；它们会先做回报完整性检查。普通/旧录制请用不带 `--rabc` 的 `train-bc`。`eval --data` 默认只评估完整留出的录制会话；至少需要两段录制，单段诊断必须显式加 `--all-data`，输出会标为 `split: all`。

`convert` 必须使用真实演示校准，且显式传入 `--allow-lossy-snn`。校准会输入 prev frame、clock、task、inventory 和 block pulse；有多个 recording 时 fidelity 使用整段留出。结果里的 `fidelity`（logit 漂移、token agreement、scope）和 `firing_rate` 是唯一可用的转换报告；没有功耗实测就不应宣称能耗收益。`train-bc` 会给 CNN 额外残差 / 第 3 层推理 / 脉冲模拟残差做零初始化，旧 `bc.pt` 不必 `--reset`。训练有光度抖动和 token-drop，预测关掉。加深加在 10Hz；30Hz 脉冲头仍 2 层、每拍 1 步 LIF。

`predict-mc` 以 30fps 抓窗口 + 状态，输出键鼠 token（含滚轮），**默认不修改任何权重**。无 `--inject` 的 `--learn` 是人类示范 CE + 冻结教师 KL；`--inject --learn` 会强制从策略分布采样，用实际 advancement/掉血与行为 shaping 得到的回报做 advantage actor-critic（TD value baseline、entropy、逐槽 categorical KL），不会把自己的 argmax 当监督标签，且拒绝 `--greedy`。GRU/Transformer 动作槽按实际采样前缀逐槽生成，采样温度、执行 token 与训练 log-prob 使用同一分布；中性转移也训练 value，瞬时 shaping 只归因给上一个动作，advancement/掉血则用默认 4 步、γ=0.9 的短程折扣反向归因。只有真正通过护栏并准备注入的动作才成为经验；护栏暂停期间不更新。后台反向只改 CPU 副本，完成后由推理线程安全发布，并优先保留稀疏奖励任务。`.plastic.pt` 同时保存模型、Adam 状态与自学习步数；`--resume-plastic` 才会显式读取它，默认不自动加载。注入拒绝 `--desktop`/`--region`，并要求 0.1.6+ Fabric sidecar 每秒回显本次随机 nonce 且证明 `singleplayer: true`、游戏窗口存在且前台；普通本机进程伪造的无 nonce UDP 包不能授权注入。任何条件失效都会释放按键并暂停学习。`play --inject` 仍然拒绝。

分步（合成冒烟仍可用 `collect --config configs/smoke.yaml`）：

```bash
python -m atomspike collect --config configs/smoke.yaml --out runs/demos.h5 --episodes 40
python -m atomspike train-bc --data runs/demos.h5 --out runs/bc.pt
# 仅在数据集回报检查通过后：
python -m atomspike train-bc --data runs/demos.h5 --out runs/bc.pt --rabc
python -m atomspike train-rl --data runs/demos.h5 --teacher runs/bc.pt --out runs/rl.pt
python -m atomspike train-peft --data runs/demos.h5 --backbone runs/rl.pt --out runs/peft.pt
python -m atomspike convert --ckpt runs/peft.pt --out runs/snn.pt --method pmsm --data runs/demos.h5 --allow-lossy-snn
python -m atomspike eval --ckpt runs/snn.pt --data runs/demos.h5 --all-data
```

合规：Minecraft 只做单机 / LAN 研究采集。`play --inject` 拒绝接到线上对战。

## 硬件角色（与文档一致）

* 96GB 卡：骨干预训练、逐游戏 LoRA、T6 转换实验

* 16GB 卡：环境评估、自博弈采样、30Hz 延迟测试

* 不做不对等双卡 DDP；7B+ 仅在大卡上 QLoRA（本 MVP 骨干约 1M 级，对标 DOOM 小模型）

## 训练基建（低算力高效率）

本实现按「单卡 16GB 闭环」设计训练基建，但所有机制不限定具体显卡型号——任何同级消费卡都能复用：

* **原子 checkpoint**：模型 + 可选优化器/调度器/GradScaler + 安全 RNG 状态；先写 `.tmp` 再 `os.replace`，加载时验证 SHA-256，并默认 `weights_only` 反序列化。BC 的最佳验证模型写入 `bc.pt`，最新训练状态写入 `bc.last.pt`；阶段内从 epoch 边界精确恢复，完成后再次运行会开启新的 WSD 微调阶段。固定输出覆盖前归档严格编号的 `*.step-N.pt`，keep-k 不会误碰 `.plastic.pt`。

* **WSD 三阶段 LR**：5% warmup / 85% stable / 10% decay。稳定期任意存档都能补一段 decay 出货，训练随时可中断。

* **Muon + AdamW 混合**：2D 权重矩阵走 Muon（Newton-Schulz 正交化动量），bias/LayerNorm/标量走 AdamW。

* **混合精度**：默认 bf16；CUDA fp16 使用动态 GradScaler（含 checkpoint 恢复），CPU fp16 自动回退 fp32；可选 `torch.compile`（默认关，配置开）。

* **梯度累积**：`grad_accum > 1` 时 micro-batch 累积后再 `optimizer.step()`，数学等价大 batch。

在 `configs/*.yaml` 里通过 `train.*` 配置：

```yaml
train:
  precision: bf16         # fp32 / bf16 / fp16（CUDA fp16 启用 GradScaler）
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
| 实时 | `eval` scheduled\_\*\_hz / `play` policy\_hz | Minecraft 仿真 30/10Hz；冒烟 30/5Hz；play 墙钟 ≈30Hz |
| 转换保真 | convert 的 fidelity / firing_rate             | 只作有损转换诊断，不代表能耗或加速 |
