<div align="center">

# vla-ur5e

A MuJoCo simulation, data collection, and VLA fine-tuning project for UR5e tabletop spatial rearrangement tasks.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Simulator-MuJoCo-111111)
![mink](https://img.shields.io/badge/Solver-mink-6A5ACD)
![OpenPI](https://img.shields.io/badge/OpenPI-PI0.5%20LoRA-2E8B57)
![GPU](https://img.shields.io/badge/GPU-%E2%89%A524GB-FF6B35)

Recognize objects, understand spatial relations, and execute language-conditioned tabletop manipulation tasks with a VLA policy.

[Setup](#setup) | [Project Overview](#project-overview) | [Data Collection](#data-collection) | [Data Processing and Replay](#data-processing-and-replay) | [Training](#training) | [Inference](#inference)

</div>

## Setup

| Part | Purpose | Requirements |
| --- | --- | --- |
| Client | MuJoCo simulation, data collection, data replay, dataset conversion, and policy server requests | Python `>= 3.12`, `mink` solver |
| Server | OpenPI training, normalization statistics, and policy server | GPU memory `>= 24 GB` |

### Client Installation

Initialize submodules:

```bash
git submodule update --init --recursive
conda env create -f environment-client.yml
conda activate vla-ur5e-client
```

OpenPI recommends installing only the lightweight `openpi-client` package on the robot/client side.

```bash
pip install --no-deps -e thirdparty/openpi/packages/openpi-client
```

### Server Installation

The server side follows OpenPI's official `uv` workflow. It is used for training and serving the policy.

```bash
cd thirdparty/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

After installation, replace the corresponding OpenPI files with the files in the `changes` directory as described in the training section.

## Project Overview

This project is a VLA training pipeline for UR5e tabletop manipulation. The goal is to make the robot not only see objects, but also ground spatial relations from language into executable actions. It builds a reproducible MuJoCo simulation environment and connects automated data collection, data replay, OpenPI fine-tuning, and online inference into a complete loop from task description to robot motion.

The project focuses on a fundamental capability for VLA-based robotic manipulation: object recognition and spatial relation understanding. The tasks include objects with different colors, shapes, and sizes, as well as spatial references such as a white square paper and a black rectangular paper. The model must identify objects such as the red cube, yellow cylinder, and cyan cuboid, then understand relations such as `left`, `right`, `up`, `down`, and `center` to complete language-conditioned tabletop rearrangement.

Compared with single-step grasping tasks, the instructions here usually contain two consecutive subtasks. For example, the robot may first place one object at the center of a paper, then place another object under a target object. This design more directly tests whether the policy understands object identity, reference objects, and spatial relations instead of memorizing fixed trajectories. Therefore, this project can serve both as a UR5e simulation data collection tool and as a compact benchmark for training and evaluating spatial-language VLA policies.

| Capability | Description |
| --- | --- |
| Object recognition | Red cube, yellow cylinder, cyan cuboid, white square paper, black rectangular paper |
| Spatial understanding | `left`, `right`, `up`, `down`, `center`, and other spatial relations |

### Task Illustration

<p align="center">
  <img src="video/task%20description.png" width="900" alt="vla-ur5e task description">
</p>

### Demo GIFs

<table>
  <tr>
    <td align="center" width="50%">
      <b>Task</b><br>
      Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid.<br><br>
      <img src="video/Move%20the%20red%20cube%20at%20the%20center%20of%20the%20black%20rectangular%20paper,%20then%20put%20the%20yellow%20cylinder%20under%20the%20cyan%20cuboid/combined_2x2.gif" width="420" alt="Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid">
    </td>
    <td align="center" width="50%">
      <b>Task</b><br>
      Transfer the cyan cuboid on the right side of the white square paper, then carry the yellow cylinder over the red cube.<br><br>
      <img src="video/Transfer%20the%20cyan%20cuboid%20on%20the%20right%20side%20of%20the%20white%20square%20paper,%20then%20carry%20the%20yellow%20cylinder%20over%20the%20red%20cube/combined_2x2.gif" width="420" alt="Transfer the cyan cuboid on the right side of the white square paper, then carry the yellow cylinder over the red cube">
    </td>
  </tr>
</table>

---

## Data Collection

The data collection entry point is `dataset_record/record.py`. It loads `description/desktop_scene.xml` by default and saves episodes in the LeRobot dataset format.

### Generate Task Descriptions

This script generates bilingual task descriptions and command sequences for automated data collection. The output is saved to `dataset_record/info/task1/task_descriptions.json`. During automated collection, `record.py` reads this file through `--task-descriptions`.

```bash
python dataset_record/task_description_generate.py
```

### Keyboard Collection

```bash
python dataset_record/record.py \
  --teleop keyboard \
  --task "Move the red cube at the center of the black rectangular paper, then put the yellow cylinder under the cyan cuboid." \
  --num-episodes 10 \
  --episode-time-s 60 \
  --overwrite
```

### Automated Collection

```bash
python dataset_record/record.py \
  --teleop autoteleop \
  --task-descriptions dataset_record/info/task1/task_descriptions.json \
  --num-episodes 10 \
  --headless \
  --overwrite
```

Common controls:

| Key | Function |
| --- | --- |
| Arrow keys / PageUp / PageDown | Move the end effector along XYZ |
| `\` | Toggle gripper open/close |
| Enter | Start / finish the current episode |
| `R` | Discard the current episode |
| Esc | Stop collection |

---

## Data Processing and Replay

### Dataset Conversion

Convert a collected LeRobot v3.0 dataset into the LeRobot v2.1 layout compatible with OpenPI.

```bash
python scripts/convert_lerobot_v30_to_v21.py \
  --root dataset_record/data/task1/bucket1_zero_completed1 \
  --output-root dataset_record/data/task1/bucket1_zero_completed1_v21 \
  --overwrite
```

### Data Replay

Replay a specified episode in MuJoCo to inspect the collected trajectory, image observations, and actions.

```bash
python dataset_record/replay.py \
  --root dataset_record/data/task1/bucket1_zero_completed1 \
  --episode 0
```

---

## Training

Training is performed inside `thirdparty/openpi`. First, replace the corresponding OpenPI files with the files in the `changes` directory.

### Replace OpenPI Files

```bash
cp changes/config.py thirdparty/openpi/src/openpi/training/config.py
cp changes/data_loader.py thirdparty/openpi/src/openpi/training/data_loader.py
cp changes/serve_policy.py thirdparty/openpi/scripts/serve_policy.py
cp changes/ur5e_policy.py thirdparty/openpi/src/openpi/policies/ur5e_policy.py
```

### Training Workflow

Enter the OpenPI directory:

```bash
cd thirdparty/openpi
```

Specify the training dataset and compute normalization statistics. Note that this uses the LeRobot dataset v2.1 layout:

```bash
export OPENPI_UR5E_DATASET_ROOT="$PWD/../../dataset_record/data/task1/bucket1_zero_completed1_v21"

uv run scripts/compute_norm_stats.py \
  --config-name pi05_ur5e_lora
```

Start PI0.5 LoRA fine-tuning:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ur5e_lora \
  --exp-name=ur5e_lora \
  --save-interval=50000 \
  --overwrite
```

The trained checkpoint is saved by default under:

```text
thirdparty/openpi/checkpoints/pi05_ur5e_lora/ur5e_lora/<step>/
```

---

## Inference

Inference has two sides: the OpenPI policy server runs the model, while the `vla-ur5e` client collects observations from MuJoCo, sends WebSocket requests, and executes returned actions.

### Start the Server

```bash
cd thirdparty/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5e_lora \
  --policy.dir=checkpoints/pi05_ur5e_lora/ur5e_lora/<step> \
  --port=8088
```

### Start the Client

```bash
python inference/client.py \
  --host <policy_server_ip> \
  --port 8088 \
  --prompt "Pick and place the yellow cylinder on the left side of the black rectangular paper." \
  --num-chunks 1000 \
  --execution-horizon 25
```

The client sends an observation containing `observation.state`, two RGB images, and the `prompt`. The server returns `actions`, and the client executes them chunk by chunk: the first three dimensions are interpreted as end-effector XYZ deltas, and the seventh dimension is thresholded as the gripper open/close command.
