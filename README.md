<div align="center">

# vla-ur5e

面向 UR5e 桌面关系重排任务的 MuJoCo 仿真、数据采集与 VLA 微调工程

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Simulator-MuJoCo-111111)
![mink](https://img.shields.io/badge/Solver-mink-6A5ACD)
![OpenPI](https://img.shields.io/badge/OpenPI-PI0.5%20LoRA-2E8B57)
![GPU](https://img.shields.io/badge/GPU-%E2%89%A524GB-FF6B35)

识别物体、理解方位，让 VLA 在仿真桌面上完成语言条件操作任务。

[项目配置](#项目配置) | [项目介绍](#项目介绍) | [数据采集](#数据采集) | [数据处理与回放](#数据处理与回放) | [训练](#训练) | [推理](#推理)

</div>

## 项目配置

| 部分 | 用途 | 要求 |
| --- | --- | --- |
| 客户端 | MuJoCo 仿真、数据采集、数据回放、数据转换、请求 policy server | Python `>= 3.12`，`mink` 求解器 |
| 服务端 | OpenPI 训练、归一化统计、启动 policy server | GPU 显存 `>= 24 GB` |

### 客户端安装

初始化子模块：

```bash
git submodule update --init --recursive
conda env create -f environment-client.yml
conda activate vla-ur5e-client
```

OpenPI 官方建议机器人端只安装轻量的 `openpi-client`。

```bash
pip install --no-deps -e thirdparty/openpi/packages/openpi-client
```

### 服务端安装

服务端使用 OpenPI 官方推荐的 `uv` 环境，负责训练和启动 policy server。

```bash
cd thirdparty/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

安装完成后，按训练章节将 `changes` 目录中的文件替换到 OpenPI 中。

## 项目介绍

此项目是一个面向 UR5e 桌面操作任务的 VLA 训练工程，目标是让机器人不只是“看见”物体，而是能够把语言里的空间关系落到真实可执行的操作上。项目用 MuJoCo 构建可复现的仿真环境，通过自动化数据采集、数据回放、OpenPI 微调和在线推理，串起从任务描述到机器人动作的完整闭环。

这个项目关注的是 VLA 机器人操作中非常基础、也非常关键的一环：物体识别与方位理解。任务中包含不同颜色、形状和尺寸的物体，以及白色方形纸片、黑色矩形纸片等空间参照物。模型需要先判断“红色方块”“黄色圆柱”“青色长方体”分别在哪里，再理解 `left`、`right`、`up`、`down`、`center` 这类相对方位，最终完成由语言驱动的桌面关系重排。

相比单步抓取任务，这里的指令通常由两个连续子任务组成，例如先把一个物体放到纸片中心，再把另一个物体放到目标物体下方。这样的设计可以更直接地检验模型是否真正理解了物体身份、参考对象和空间关系，而不是只记住固定轨迹。因此，此项目既可以作为 UR5e 仿真数据采集工具，也可以作为训练和验证空间语言理解型 VLA policy 的小型基准工程。

| 能力 | 内容 |
| --- | --- |
| 物体识别 | 红色方块、黄色圆柱、青色长方体、白色方形纸片、黑色矩形纸片 |
| 方位理解 | `left`、`right`、`up`、`down`、`center` 等空间关系 |

### 任务示意图

<p align="center">
  <img src="video/task%20description.png" width="900" alt="vla-ur5e task description">
</p>

### 演示 GIF

<table>
  <tr>
    <td align="center" width="50%">
      <b>中文</b><br>
      将红色方块移动到黑色矩形纸片中心，然后将黄色圆柱放到青色长方体下方。<br><br>
      <b>English</b><br>
      Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid.<br><br>
      <img src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/combined_2x2.gif" width="420" alt="Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid">
    </td>
    <td align="center" width="50%">
      <b>中文</b><br>
      将青色长方体移动到白色方形纸片右侧，然后将黄色圆柱移动到红色方块上方。<br><br>
      <b>English</b><br>
      Transfer the cyan cuboid on the right side of the white square paper, then carry the yellow cylinder over the red cube.<br><br>
      <img src="video/Transfer%20the%20cyan%20cuboid%20on%20the%20right%20side%20of%20the%20white%20square%20paper,%20then%20carry%20the%20yellow%20cylinder%20over%20the%20red%20cube/combined_2x2.gif" width="420" alt="Transfer the cyan cuboid on the right side of the white square paper, then carry the yellow cylinder over the red cube">
    </td>
  </tr>
</table>

---

## 数据采集

数据采集入口是 `dataset_record/record.py`，默认加载 `description/desktop_scene.xml`，并保存为 LeRobot 格式数据集。

### 生成任务描述

用于批量生成中英文任务描述和自动化采集所需的命令序列，输出到 `dataset_record/info/task1/task_descriptions.json`。自动化采集时，`record.py` 会通过 `--task-descriptions` 读取该文件。

```bash
python dataset_record/task_description_generate.py
```

### 键盘采集

```bash
python dataset_record/record.py \
  --teleop keyboard \
  --task "Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid." \
  --num-episodes 10 \
  --episode-time-s 60 \
  --overwrite
```

### 自动化采集

```bash
python dataset_record/record.py \
  --teleop autoteleop \
  --task-descriptions dataset_record/info/task1/task_descriptions.json \
  --num-episodes 10 \
  --headless \
  --overwrite
```

常用控制：

| 按键 | 功能 |
| --- | --- |
| 方向键 / PageUp / PageDown | 控制末端 XYZ 移动 |
| `\` | 切换夹爪开合 |
| Enter | 开始 / 结束当前 episode |
| `R` | 丢弃当前 episode |
| Esc | 停止采集 |

---

## 数据处理与回放

### 数据集转换

将采集得到的 LeRobot v3.0 数据集转换为 OpenPI 兼容的 LeRobot v2.1 布局。

```bash
python scripts/convert_lerobot_v30_to_v21.py \
  --root dataset_record/data/task1/bucket1_zero_completed1 \
  --output-root dataset_record/data/task1/bucket1_zero_completed1_v21 \
  --overwrite
```

### 数据回放

在 MuJoCo 中回放指定 episode，便于检查采集轨迹、图像观测和动作是否正常。

```bash
python dataset_record/replay.py \
  --root dataset_record/data/task1/bucket1_zero_completed1 \
  --episode 0
```

---

## 训练

训练在 `thirdparty/openpi` 中完成。先用 `changes` 目录下的文件替换 OpenPI 中对应文件。

### 替换 OpenPI 文件

```bash
cp changes/config.py thirdparty/openpi/src/openpi/training/config.py
cp changes/data_loader.py thirdparty/openpi/src/openpi/training/data_loader.py
cp changes/serve_policy.py thirdparty/openpi/scripts/serve_policy.py
cp changes/ur5e_policy.py thirdparty/openpi/src/openpi/policies/ur5e_policy.py
```

### 训练流程

进入 OpenPI：

```bash
cd thirdparty/openpi
```

指定训练数据集并计算归一化统计，注意这里使用lerobot_dataset v21格式数据集：

```bash
export OPENPI_UR5E_DATASET_ROOT="$PWD/../../dataset_record/data/task1/bucket1_zero_completed1_v21"

uv run scripts/compute_norm_stats.py \
  --config-name pi05_ur5e_lora
```

启动 PI0.5 LoRA 微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ur5e_lora \
  --exp-name=ur5e_lora \
  --save-interval=50000 \
  --overwrite
```

训练完成后 checkpoint 默认保存在：

```text
thirdparty/openpi/checkpoints/pi05_ur5e_lora/ur5e_lora/<step>/
```

---

## 推理

推理分为两端：OpenPI policy server 负责跑模型，`vla-ur5e` client 负责从 MuJoCo 采集观测、发送 WebSocket 请求并执行返回动作。

### 启动服务端

```bash
cd thirdparty/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5e_lora \
  --policy.dir=checkpoints/pi05_ur5e_lora/ur5e_lora/<step> \
  --port=8088
```

### 启动客户端

```bash
python inference/client.py \
  --host <policy_server_ip> \
  --port 8088 \
  --prompt "Pick and place the yellow cylinder on the left side of the black rectangular paper." \
  --num-chunks 1000 \
  --execution-horizon 25
```

客户端发送的 observation 包含 `observation.state`、两路 RGB 图像和 `prompt`。服务端返回 `actions`，客户端按 chunk 执行动作：前三维作为末端 XYZ 增量，第七维按阈值解释为夹爪开合。
