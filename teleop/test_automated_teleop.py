from __future__ import annotations


import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from dataset_record.config import AutomatedTeleopConfig, RecordConfig
from env import LabSimMujocoEnv
from teleop.automated_teleop import (
    AutomatedTeleop,
    PickPlaceCommand,
    parse_task,
)


class FakeLabSimEnv:
    def __init__(self) -> None:
        self.site_positions = {
            "red_cube_center_site": np.asarray([0.1, 0.2, 0.715]),
            "yellow_cylinder_left_site": np.asarray([-0.1, 0.3, 0.72]),
            "yellow_cylinder_center_site": np.asarray([0.0, 0.3, 0.72]),
        }
        self.calls: list[tuple] = []

    def get_site_position(self, site_name: str) -> np.ndarray:
        self.calls.append(("get_site_position", site_name))
        return self.site_positions[site_name].copy()

    def move_pinch_to(
        self,
        target_position: np.ndarray,
        *,
        max_speed_m_s: float,
        position_tolerance_m: float,
        max_motion_time_s: float,
        viewer=None,
        step_callback=None,
    ) -> None:
        self.calls.append(
            (
                "move_pinch_to",
                np.asarray(target_position).copy(),
                max_speed_m_s,
                position_tolerance_m,
                max_motion_time_s,
                viewer,
            )
        )
        if step_callback is not None:
            step_callback()

    def set_gripper(
        self,
        closed: bool,
        *,
        settle_time_s: float = 0.0,
        viewer=None,
        step_callback=None,
    ) -> None:
        self.calls.append(("set_gripper", closed, settle_time_s, viewer))
        if step_callback is not None:
            step_callback()

    def reset(self) -> None:
        raise AssertionError("AutomatedTeleop must not reset the external env")

    def close(self) -> None:
        raise AssertionError("AutomatedTeleop must not close the external env")


class ParseTaskTest(unittest.TestCase):
    def test_parses_multiple_instructions(self) -> None:
        description = (
            "将红色方块平移到黄色圆柱的左侧，"
            "然后将青色长方体放置在红色方块的上方"
        )

        commands = parse_task(description)

        self.assertEqual(
            commands,
            [
                PickPlaceCommand("red_cube", "yellow_cylinder", "left"),
                PickPlaceCommand("cyan_cuboid", "red_cube", "up"),
            ],
        )

    def test_parses_new_sheets_as_targets(self) -> None:
        description = (
            "将红色方块移动到白色正方形纸片的左侧，"
            "然后将黄色圆柱放置到黑色纸片的上方"
        )

        commands = parse_task(description)

        self.assertEqual(
            commands,
            [
                PickPlaceCommand("red_cube", "white_square_sheet", "left"),
                PickPlaceCommand(
                    "yellow_cylinder", "black_rectangular_sheet", "up"
                ),
            ],
        )

    def test_parses_english_instructions(self) -> None:
        description = (
            "Move the red block to the left of the white paper. "
            "Then place the yellow cylinder on the right side of the black card."
        )

        commands = parse_task(description)

        self.assertEqual(
            commands,
            [
                PickPlaceCommand("red_cube", "white_square_sheet", "left"),
                PickPlaceCommand(
                    "yellow_cylinder", "black_rectangular_sheet", "right"
                ),
            ],
        )

    def test_parses_extended_chinese_aliases_and_verbs(self) -> None:
        commands = parse_task("把红色小方块拿到白色卡片的左手边")

        self.assertEqual(
            commands,
            [PickPlaceCommand("red_cube", "white_square_sheet", "left")],
        )

    def test_duplicate_bilingual_instruction_is_returned_once(self) -> None:
        description = (
            "将红方块移动到白纸左侧。"
            "Move the red cube to the left of the white paper."
        )

        self.assertEqual(
            parse_task(description),
            [PickPlaceCommand("red_cube", "white_square_sheet", "left")],
        )

    def test_sheet_cannot_be_used_as_source_object(self) -> None:
        description = (
            "将白色纸片移动到红色方块的左侧，"
            "然后将黄色圆柱放置到黑色纸片的上方"
        )

        with self.assertRaisesRegex(ValueError, "object is not graspable"):
            parse_task(description)

    def test_single_instruction_is_supported(self) -> None:
        commands = parse_task("将红色方块移动到黄色圆柱的右侧")

        self.assertEqual(
            commands,
            [PickPlaceCommand("red_cube", "yellow_cylinder", "right")],
        )

    def test_description_without_instruction_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pick-place instruction"):
            parse_task("")


class AutomatedTeleopTest(unittest.TestCase):
    def test_step_callback_covers_gripper_and_motion_steps(self) -> None:
        env = FakeLabSimEnv()
        callback_count = 0

        def on_step() -> None:
            nonlocal callback_count
            callback_count += 1

        AutomatedTeleop(env).run(
            [PickPlaceCommand("red_cube", "yellow_cylinder", "left")],
            step_callback=on_step,
        )

        self.assertEqual(callback_count, 10)

    def test_pick_place_uses_external_env_in_expected_order(self) -> None:
        env = FakeLabSimEnv()
        viewer = object()
        motion_config = AutomatedTeleopConfig(
            hover_height_m=0.12,
            move_speed_m_s=0.05,
            position_tolerance_m=0.003,
            max_motion_time_s=4.0,
            gripper_settle_time_s=0.25,
        )
        automated_teleop = AutomatedTeleop(env, motion_config)

        automated_teleop.run(
            [PickPlaceCommand("red_cube", "yellow_cylinder", "left")],
            viewer=viewer,
        )

        gripper_calls = [call for call in env.calls if call[0] == "set_gripper"]
        self.assertEqual(
            [call[1] for call in gripper_calls],
            [False, False, True, False],
        )
        for _, _, settle_time_s, received_viewer in gripper_calls:
            self.assertEqual(settle_time_s, 0.25)
            self.assertIs(received_viewer, viewer)

        move_calls = [call for call in env.calls if call[0] == "move_pinch_to"]
        expected_targets = [
            [0.1, 0.2, 0.835],
            [0.1, 0.2, 0.715],
            [0.1, 0.2, 0.835],
            [-0.1, 0.3, 0.85],
            [-0.1, 0.3, 0.73],
            [-0.1, 0.3, 0.85],
        ]
        self.assertEqual(len(move_calls), len(expected_targets))
        for call, expected_target in zip(move_calls, expected_targets, strict=True):
            _, target, speed, tolerance, timeout, received_viewer = call
            np.testing.assert_allclose(target, expected_target)
            self.assertEqual(speed, 0.05)
            self.assertEqual(tolerance, 0.003)
            self.assertEqual(timeout, 4.0)
            self.assertIs(received_viewer, viewer)

        self.assertEqual(
            [call for call in env.calls if call[0] == "get_site_position"],
            [
                ("get_site_position", "red_cube_center_site"),
                ("get_site_position", "yellow_cylinder_left_site"),
                ("get_site_position", "yellow_cylinder_center_site"),
            ],
        )


class MujocoSheetModelTest(unittest.TestCase):
    def test_hold_invokes_step_callback_each_control_step(self) -> None:
        env = LabSimMujocoEnv(RecordConfig())
        callback_count = 0

        def on_step() -> None:
            nonlocal callback_count
            callback_count += 1

        try:
            env.reset()
            env.hold(env.control_dt * 2, step_callback=on_step)
        finally:
            env.close()

        self.assertEqual(callback_count, 2)

    def test_new_sheet_sites_exist_in_real_mujoco_model(self) -> None:
        env = LabSimMujocoEnv(RecordConfig())
        try:
            self.assertIsNone(env.reset())
            for object_name in (
                "white_square_sheet",
                "black_rectangular_sheet",
            ):
                for position in ("center", "up", "down", "left", "right"):
                    site_position = env.get_site_position(
                        f"{object_name}_{position}_site"
                    )
                    self.assertEqual(site_position.shape, (3,))
                    self.assertTrue(np.isfinite(site_position).all())
        finally:
            env.close()


def run_visual_mujoco_test() -> None:
    description = (
        "将红色方块移动到白色正方形纸片的中心，"
        "然后将黄色圆柱放置到红色方块的右侧"
    )
    commands = parse_task(description)

    env = LabSimMujocoEnv(RecordConfig())
    env.reset()
    automated_teleop = AutomatedTeleop(env)

    print(f"Visual test task: {description}")
    for command in commands:
        print(
            f"Parsed command: source={command.source_object}, "
            f"target={command.target_object}, position={command.target_position}"
        )

    try:
        with env.viewer_context(headless=False) as viewer:
            env.hold(1.0, viewer=viewer)
            automated_teleop.run(commands, viewer=viewer)
            env.hold(3.0, viewer=viewer)
    finally:
        env.close()


def run_task1_descriptions_visual_test() -> None:
    descriptions_path = (
        Path(__file__).resolve().parents[1]
        / "dataset_record"
        / "info"
        / "task1"
        / "task_descriptions.json"
    )
    payload = json.loads(descriptions_path.read_text(encoding="utf-8"))
    descriptions = payload["descriptions"]

    env = LabSimMujocoEnv(RecordConfig())
    automated_teleop = AutomatedTeleop(env)

    try:
        with env.viewer_context(headless=False) as viewer:
            for index, item in enumerate(descriptions, start=1):
                chinese_description = item["chinese"]
                english_description = item["english"]
                print(f"\n[{index}/{len(descriptions)}] {item['id']}", flush=True)
                print(f"Chinese: {chinese_description}", flush=True)
                print(f"English: {english_description}", flush=True)

                commands = parse_task(english_description)
                expected_commands = [
                    PickPlaceCommand(**command) for command in item["commands"]
                ]
                if commands != expected_commands:
                    raise ValueError(
                        f"English description does not match commands for {item['id']}: "
                        f"parsed={commands}, expected={expected_commands}"
                    )

                env.reset()
                env.hold(0.5, viewer=viewer)
                automated_teleop.run(commands, viewer=viewer)
                env.hold(0.5, viewer=viewer)
    finally:
        env.close()


if __name__ == "__main__":
    run_task1_descriptions_visual_test()
    #run_visual_mujoco_test
