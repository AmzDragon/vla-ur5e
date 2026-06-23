# lab_sim

MuJoCo UR5e desktop manipulation simulation for keyboard teleoperation,
scripted pick-and-place data collection, and OpenPI-style remote inference.

## Project Layout

- `description/desktop_scene.xml` is the default MuJoCo scene.
- `env/` contains the MuJoCo environment wrapper.
- `teleop/` contains keyboard teleoperation, IK, and scripted pick-place logic.
- `dataset_record/` records LeRobot-format datasets.
- `inference/` contains a small OpenPI websocket inference client.
- `scripts/run_example.py` starts an interactive MuJoCo viewer.

## Quick Start

Install the Python dependencies used by the code, including `mujoco`, `numpy`,
`mink`, and the vendored packages under `thirdparty/` when using dataset or
inference features.

```bash
python scripts/run_example.py
python dataset_record/record.py --teleop keyboard --num-episodes 1
python dataset_record/record.py --teleop autoteleop --headless --num-episodes 1
```

The default scene path is configured in `dataset_record/config.py` as
`description/desktop_scene.xml`.

## Description Files

The current code loads `description/desktop_scene.xml` by default. The older
`description/scene.xml` file is kept as an optional/legacy scene and is not
loaded by the current environment configuration.

<details>
<summary>Files used by the default description scene</summary>

XML files:

- `description/desktop_scene.xml`
- `description/object/black_rectangular_sheet.xml`
- `description/object/cyan_cuboid.xml`
- `description/object/red_cube.xml`
- `description/object/white_square_sheet.xml`
- `description/object/yellow_cylinder.xml`
- `description/robotiq_2f85/robotiq2f85.xml`
- `description/robotiq_2f85/robotiq2f85_globals.xml`
- `description/ur5e/ur5e.xml`
- `description/ur5e/ur5e_globals.xml`

Mesh/assets:

- `description/assets/base.stl`
- `description/assets/base_0.obj`
- `description/assets/base_1.obj`
- `description/assets/base_platform.stl`
- `description/assets/camera_adapter.stl`
- `description/assets/forearm_0.obj`
- `description/assets/forearm_1.obj`
- `description/assets/forearm_2.obj`
- `description/assets/forearm_3.obj`
- `description/assets/mtc_ur3510_ur_toolchanger_collision.stl`
- `description/assets/robotiq_2f85/base.stl`
- `description/assets/robotiq_2f85/finger_link.stl`
- `description/assets/robotiq_2f85/finger_tip_link.stl`
- `description/assets/robotiq_2f85/inner_knuckle_link.stl`
- `description/assets/robotiq_2f85/knuckle_link.stl`
- `description/assets/robotiq_2f85/pad.stl`
- `description/assets/robotiq_2f85/silicone_pad.stl`
- `description/assets/shoulder_0.obj`
- `description/assets/shoulder_1.obj`
- `description/assets/shoulder_2.obj`
- `description/assets/upperarm_0.obj`
- `description/assets/upperarm_1.obj`
- `description/assets/upperarm_2.obj`
- `description/assets/upperarm_3.obj`
- `description/assets/wrist1_0.obj`
- `description/assets/wrist1_1.obj`
- `description/assets/wrist1_2.obj`
- `description/assets/wrist2_0.obj`
- `description/assets/wrist2_1.obj`
- `description/assets/wrist2_2.obj`
- `description/assets/wrist3.obj`
- `description/object/meshes/plastic_crate.obj`

</details>

## License

Project code is released under the BSD 3-Clause License. Third-party assets and
vendored code retain their own licenses; keep the nested `LICENSE` files in
`description/` and `thirdparty/` with those files.
