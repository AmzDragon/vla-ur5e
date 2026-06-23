from pathlib import Path
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
import numpy as np

from teleop.keyboard_teleop import KeyboardTeleop
from teleop.mink_ik_solver import MinkIKSolver


ROBOT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_NAME = "robotiq_85_left"
GRIPPER_JOINT_CANDIDATES = (
    GRIPPER_NAME,
    f"{GRIPPER_NAME}_knuckle_joint",
)
GRIPPER_ACTUATOR_NAME = "robotiq_85_left_knuckle_joint"
DROID_JOINT_POSITION_DIM = 7
DROID_CAMERA_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)
CONTROL_HZ = 30.0
END_EFF_SPEED = 0.1  # m/s
THETAT = END_EFF_SPEED / CONTROL_HZ
GRIPPER_OPEN_CTRL = 0.0
GRIPPER_CLOSED_CTRL = 0.8
INITIAL_CTRL = np.array([0.0, -1.76, 2.0, 4.34, -1.5, 0.0, 0.0], dtype=np.float64)


def get_env_state(env, cameras: Sequence[str], prompt: str, image_size: int = 224) -> dict:
    """Build a DROID-style openpi observation from a MuJoCo env/model-data pair."""
    model, data = _get_model_data(env)
    if len(cameras) != len(DROID_CAMERA_KEYS):
        raise ValueError(
            "DROID policy expects exactly two cameras: "
            "[exterior_camera_name, wrist_camera_name]"
        )

    robot_joint_position = np.asarray(
        [_get_joint_qpos(model, data, name) for name in ROBOT_JOINT_NAMES],
        dtype=np.float32,
    )
    joint_position = _pad_to_droid_joint_position(robot_joint_position)
    gripper_position = np.asarray([_get_gripper_qpos(model, data)], dtype=np.float32)

    observation = {
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    for camera, observation_key in zip(cameras, DROID_CAMERA_KEYS):
        observation[observation_key] = _render_camera(
            model,
            data,
            camera,
            image_size=image_size,
        )
    return observation


def _get_model_data(env):
    if isinstance(env, tuple) and len(env) == 2:
        return env
    if isinstance(env, dict) and "model" in env and "data" in env:
        return env["model"], env["data"]
    if hasattr(env, "model") and hasattr(env, "data"):
        return env.model, env.data
    if (
        hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "model")
        and hasattr(env.unwrapped, "data")
    ):
        return env.unwrapped.model, env.unwrapped.data
    raise TypeError("env must expose MuJoCo model/data or be a (model, data) tuple")


def _get_joint_qpos(model, data, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"joint not found in model: {joint_name}")
    qpos_addr = model.jnt_qposadr[joint_id]
    return float(data.qpos[qpos_addr])


def _get_gripper_qpos(model, data) -> float:
    for joint_name in GRIPPER_JOINT_CANDIDATES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_addr = model.jnt_qposadr[joint_id]
            return float(data.qpos[qpos_addr])
    candidates = ", ".join(GRIPPER_JOINT_CANDIDATES)
    raise KeyError(f"gripper joint not found in model. Tried: {candidates}")


def _pad_to_droid_joint_position(joint_position: np.ndarray) -> np.ndarray:
    if joint_position.shape[-1] > DROID_JOINT_POSITION_DIM:
        raise ValueError(
            f"DROID joint_position expects at most {DROID_JOINT_POSITION_DIM} "
            f"values, got {joint_position.shape[-1]}"
        )
    if joint_position.shape[-1] == DROID_JOINT_POSITION_DIM:
        return joint_position
    pad_width = DROID_JOINT_POSITION_DIM - joint_position.shape[-1]
    return np.pad(joint_position, (0, pad_width), constant_values=0.0).astype(
        np.float32,
        copy=False,
    )


def _render_camera(model, data, camera: str, image_size: int) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=image_size, width=image_size)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render().astype(np.uint8, copy=False).copy()
    finally:
        renderer.close()


def _actuator_qpos_addr(model, actuator_id: int) -> int:
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        raise ValueError(f"actuator does not target a joint: {actuator_name}")
    return int(model.jnt_qposadr[joint_id])


def _actuator_id(model, actuator_name: str) -> int:
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )
    if actuator_id < 0:
        raise KeyError(f"actuator not found in model: {actuator_name}")
    return int(actuator_id)


def _initialize_from_ctrl(model, data, ctrl: np.ndarray) -> None:
    if ctrl.shape != (model.nu,):
        raise ValueError(f"expected ctrl shape {(model.nu,)}, got {ctrl.shape}")

    data.ctrl[:] = ctrl
    for actuator_id, value in enumerate(ctrl):
        data.qpos[_actuator_qpos_addr(model, actuator_id)] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _sync_ctrl_from_qpos(model, data, qpos: np.ndarray, actuator_ids) -> None:
    for actuator_id in actuator_ids:
        qpos_addr = _actuator_qpos_addr(model, actuator_id)
        value = float(qpos[qpos_addr])
        if model.actuator_ctrllimited[actuator_id]:
            low, high = model.actuator_ctrlrange[actuator_id]
            value = float(np.clip(value, low, high))
        data.ctrl[actuator_id] = value


def main() -> None:
    xml_path = Path(__file__).resolve().parents[1] / "description" / "desktop_scene.xml"

    solver = MinkIKSolver.from_xml_path(
        xml_path,
        frame_name="attachment_site",
        frame_type="site",
        fps=CONTROL_HZ,
        position_cost=1.0,
        orientation_cost=1.0,
        gain=0.2,
        solver="daqp",
        solve_ik_kwargs={"damping": 1e-4},
    )
    model = solver.configuration.model
    arm_actuator_ids = [_actuator_id(model, name) for name in ROBOT_JOINT_NAMES]
    gripper_actuator_id = _actuator_id(model, GRIPPER_ACTUATOR_NAME)
    gripper_midpoint = 0.5 * (GRIPPER_OPEN_CTRL + GRIPPER_CLOSED_CTRL)
    teleop = KeyboardTeleop(
        hz=CONTROL_HZ,
        gripper_closed=INITIAL_CTRL[gripper_actuator_id] > gripper_midpoint,
    )

    data = mujoco.MjData(model)
    _initialize_from_ctrl(model, data, INITIAL_CTRL)
    solver.configuration.update(data.qpos.copy())
    solver.reset_target_to_current()

    control_dt = 1.0 / CONTROL_HZ
    next_tick = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_xyz = teleop.update()
            solver.configuration.update(data.qpos.copy())
            solver.step(step_xyz, scale=THETAT)
            _sync_ctrl_from_qpos(model, data, solver.qpos(), arm_actuator_ids)
            data.ctrl[gripper_actuator_id] = teleop.get_gripper_command(
                open_value=GRIPPER_OPEN_CTRL,
                closed_value=GRIPPER_CLOSED_CTRL,
            )

            sim_time_target = data.time + control_dt
            while data.time < sim_time_target:
                mujoco.mj_step(model, data)
            viewer.sync()

            next_tick += control_dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))


if __name__ == "__main__":
    main()
