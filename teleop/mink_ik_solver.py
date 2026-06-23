from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mink import Configuration, FrameTask, SE3, solve_ik

XYZ = np.ndarray | list[float] | tuple[float, float, float]


@dataclass
class MinkIKSolver:
    """Small wrapper around mink FrameTask IK for Cartesian teleoperation.

    The wrapper owns a mink ``Configuration`` and a single ``FrameTask``.  A
    keyboard step can be applied to the task target with ``step``; then the
    wrapper solves one IK frame and integrates the configuration in place.
    """

    model: Any
    frame_name: str
    frame_type: str = "site"
    fps: float = 30.0
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    gain: float = 1.0
    solver: str = "daqp"
    keyframe: str | None = None
    solve_ik_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.fps
        self.configuration = Configuration(self.model)
        if self.keyframe is not None:
            self.configuration.update_from_keyframe(self.keyframe)

        self.task = FrameTask(
            frame_name=self.frame_name,
            frame_type=self.frame_type,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            gain=self.gain,
        )
        self.target = self.get_current_pose()
        self.task.set_target(self.target)

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str | Path,
        *,
        frame_name: str,
        frame_type: str = "site",
        fps: float = 30.0,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        gain: float = 1.0,
        solver: str = "daqp",
        keyframe: str | None = None,
        solve_ik_kwargs: dict[str, Any] | None = None,
    ) -> "MinkIKSolver":
        """Load a MuJoCo model from XML and create the IK wrapper."""
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        return cls(
            model=model,
            frame_name=frame_name,
            frame_type=frame_type,
            fps=fps,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=gain,
            solver=solver,
            keyframe=keyframe,
            solve_ik_kwargs=solve_ik_kwargs or {},
        )

    @staticmethod
    def convergence_gain(
        *,
        duration: float = 2.0,
        fps: float = 60.0,
        final_error_ratio: float = 0.01,
    ) -> float:
        """Return the exponential task gain used by the mink example."""
        n_frames = int(duration * fps)
        return 1.0 - final_error_ratio ** (1.0 / n_frames)

    def get_current_pose(self) -> Any:
        """Return the controlled frame pose in the world frame."""
        return self.configuration.get_transform_frame_to_world(
            self.frame_name,
            self.frame_type,
        )

    def reset_target_to_current(self) -> Any:
        """Set the IK target to the current end-effector pose."""
        self.target = self.get_current_pose()
        self.task.set_target(self.target)
        return self.target

    def set_target(self, target: Any) -> None:
        """Set an explicit mink SE3 target."""
        self.target = target
        self.task.set_target(self.target)

    def set_target_from_current(
        self,
        translation: XYZ,
        *,
        rotation: Any | None = None,
    ) -> Any:
        """Set a target from the current pose plus a world-frame offset."""
        target = SE3.from_translation(np.asarray(translation)) @ self.get_current_pose()
        if rotation is not None:
            target = target @ SE3.from_rotation(rotation)
        self.set_target(target)
        return self.target

    def translate_target(
        self,
        delta_xyz: XYZ,
        *,
        scale: float = 1.0,
        in_world_frame: bool = True,
    ) -> Any:    
        """Move the current target by ``delta_xyz * scale``."""
        delta = SE3.from_translation(np.asarray(delta_xyz) * scale)
        if in_world_frame:
            self.target = delta @ self.target
        else:
            self.target = self.target @ delta
        self.task.set_target(self.target)
        return self.target

    def solve_step(self, *, dt: float | None = None) -> np.ndarray:
        """Solve IK once and integrate the configuration in place."""
        step_dt = self.dt if dt is None else dt
        vel = solve_ik(
            self.configuration,
            [self.task],
            step_dt,
            self.solver,
            **self.solve_ik_kwargs,
        )
        self.configuration.integrate_inplace(vel, step_dt)
        return vel

    def step(
        self,
        delta_xyz: XYZ | None = None,
        *,
        scale: float = 1.0,
        dt: float | None = None,
        in_world_frame: bool = True,
    ) -> np.ndarray:
        """Optionally move the target, then solve and integrate one IK step."""
        if delta_xyz is not None:
            self.translate_target(
                delta_xyz,
                scale=scale,
                in_world_frame=in_world_frame,
            )
        return self.solve_step(dt=dt)

    def qpos(self) -> np.ndarray:
        """Return a copy of the current generalized positions."""
        return self.configuration.q.copy()

    def position_error(self) -> float:
        """Return the Euclidean position error between current pose and target."""
        current = self.get_current_pose()
        return float(np.linalg.norm(current.translation() - self.target.translation()))
