<div align="center">

# lab_sim

面向 UR5e 桌面操作的 MuJoCo 仿真、数据采集与 VLA 微调工程

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Simulator-MuJoCo-111111)
![OpenPI](https://img.shields.io/badge/OpenPI-PI0.5%20LoRA-2E8B57)
![LeRobot](https://img.shields.io/badge/Dataset-LeRobot-FFB000)

感知物体、理解方位、输出动作，让 VLA 在仿真桌面上完成可执行的 pick-and-place 任务。

[项目介绍](#项目介绍) | [数据采集](#数据采集) | [训练](#训练) | [推理](#推理)

</div>

---

## 项目介绍

`lab_sim` 用 MuJoCo 搭建 UR5e + Robotiq 2F-85 桌面操作环境，围绕“识别物体”和“理解相对方位”训练 VLA policy。模型输入来自外部相机、腕部相机、末端状态和语言指令；模型输出为连续动作 chunk，用于驱动末端移动和夹爪开合。

| 能力 | 内容 |
| --- | --- |
| 物体识别 | 红色方块、黄色圆柱、青色长方体、白色方形纸片、黑色矩形纸片 |
| 方位理解 | `left`、`right`、`up`、`down`、`center` 等空间关系 |
| 观测输入 | 末端 7 维状态 + 外部相机 RGB + 腕部相机 RGB + 语言 prompt |
| 动作输出 | 7 维动作：末端位姿增量 + 夹爪开合 |
| 训练方案 | OpenPI / PI0.5 LoRA 微调 |

### 演示视频

两个任务各包含 4 段前视角演示视频。若 GitHub 页面没有直接播放视频，可点击每个 episode 下方的链接打开。

<details open>
<summary><b>Task 1: Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid</b></summary>

<table>
  <tr>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_00.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_00.mp4">Episode 00</a>
    </td>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_01.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_01.mp4">Episode 01</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_02.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_02.mp4">Episode 02</a>
    </td>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_03.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/front_camera_episode_03.mp4">Episode 03</a>
    </td>
  </tr>
</table>

</details>

<details open>
<summary><b>Task 2: Move the red cube under the cyan cuboid, then put the yellow cylinder on the left side of the black rectangular paper</b></summary>

<table>
  <tr>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_00.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_00.mp4">Episode 00</a>
    </td>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_01.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_01.mp4">Episode 01</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_02.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_02.mp4">Episode 02</a>
    </td>
    <td align="center">
      <video src="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_03.mp4" controls width="320"></video>
      <br><a href="video/Move%20the%20red%20cube%20under%20the%20cyan%20cuboid,%20then%20put%20the%20yellow%20cylinder%20on%20the%20left%20side%20of%20the%20black%20rectangular%20paper/front_camera_episode_03.mp4">Episode 03</a>
    </td>
  </tr>
</table>

</details>

---

## 数据采集

数据采集入口是 `dataset_record/record.py`，默认加载 `description/desktop_scene.xml`，并保存为 LeRobot 格式数据集。

### 采集命令

键盘遥操作采集：

```bash
python dataset_record/record.py \
  --teleop keyboard \
  --num-episodes 10 \
  --episode-time-s 60 \
  --overwrite
```

自动脚本采集：

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

### 数据字段

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `observation.state` | `(7,)` | `[x, y, z, roll, pitch, yaw, gripper_closed]` |
| `action` | `(7,)` | `[theta_x, theta_y, theta_z, delta_roll, delta_pitch, delta_yaw, gripper_closed]` |
| `observation.images.exterior_image_1_left` | `(224, 224, 3)` | 外部相机 RGB 图像 |
| `observation.images.wrist_image_left` | `(224, 224, 3)` | 腕部相机 RGB 图像 |
| `task` | text | 语言任务描述 |

默认输出目录在 `dataset_record/config.py` 的 `RecordConfig.dataset_root` 中配置。训练前建议把最终数据集路径显式传给 OpenPI，避免默认路径不一致。

---

## 训练

训练在 `thirdparty/openpi` 中完成。当前工程已经为 UR5e 数据接入 OpenPI 增加了专门的 policy、data config 和 LoRA train config。

### OpenPI 需要修改的代码

| 文件 | 需要的内容 |
| --- | --- |
| `src/openpi/policies/ur5e_policy.py` | 定义 `UR5Inputs` / `UR5Outputs`，把 lab_sim 的状态、双相机图像、prompt 转成 PI0.5 输入格式 |
| `src/openpi/training/config.py` | 引入 `ur5e_policy`，增加 `DataConfig.dataset_root`、`LeRobotUR5eDataConfig`、`pi05_ur5e_lora` |
| `src/openpi/training/data_loader.py` | 创建 `LeRobotDatasetMetadata` / `LeRobotDataset` 时传入 `root=data_config.dataset_root`，支持读取本地数据集 |
| `scripts/convert_lerobot_v30_to_v21.py` | 如果当前 OpenPI 读取器不兼容 LeRobot v3.0，用它转换为 v2.1 布局 |

`UR5Inputs` 的图像槽位映射需要和推理端保持一致：

```python
image = {
    "base_0_rgb": exterior_image,
    "left_wrist_0_rgb": zeros,
    "right_wrist_0_rgb": wrist_image,
}
image_mask = {
    "base_0_rgb": True,
    "left_wrist_0_rgb": False,
    "right_wrist_0_rgb": True,
}
```

### 训练流程

进入 OpenPI：

```bash
cd thirdparty/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

如果需要转换 LeRobot 版本：

```bash
uv run scripts/convert_lerobot_v30_to_v21.py \
  --root ../../dataset_record/data/task1/bucket1_zero_completed1 \
  --output-root ../../dataset_record/data/task1/bucket1_zero_completed1_v21 \
  --overwrite
```

指定训练数据集并计算归一化统计：

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
  --overwrite
```

训练完成后 checkpoint 默认保存在：

```text
thirdparty/openpi/checkpoints/pi05_ur5e_lora/ur5e_lora/<step>/
```

---

## 推理

推理分为两端：OpenPI policy server 负责跑模型，`lab_sim` client 负责从 MuJoCo 采集观测、发送 WebSocket 请求并执行返回动作。

### 安装依赖

OpenPI 服务端：

```bash
cd thirdparty/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

lab_sim 客户端：

```bash
pip install mujoco mink daqp numpy pillow websockets msgpack dm-tree
pip install -e thirdparty/openpi/packages/openpi-client
```

如果还要在客户端侧采集或回放 LeRobot 数据，再安装本仓库 vendored 的 LeRobot：

```bash
pip install -e thirdparty/lerobot
```

### 启动服务端

```bash
cd thirdparty/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5e_lora \
  --policy.dir=checkpoints/pi05_ur5e_lora/ur5e_lora/<step> \
  --port=8088
```

### 启动仿真推理

回到项目根目录：

```bash
python inference/client.py \
  --host <policy_server_ip> \
  --port 8088 \
  --prompt "Pick and place the yellow cylinder on the left side of the black rectangular paper." \
  --num-chunks 1000 \
  --execution-horizon 25
```

客户端发送的 observation 包含 `observation.state`、两路 RGB 图像和 `prompt`。服务端返回 `actions`，客户端按 chunk 执行动作：前三维作为末端 XYZ 增量，第七维按阈值解释为夹爪开合。
