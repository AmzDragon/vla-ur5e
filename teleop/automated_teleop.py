from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from dataset_record.config import AutomatedTeleopConfig
import numpy as np

from env import LabSimMujocoEnv


OBJECT_ALIASES = {
    "white_square_sheet": (
        "white_square_sheet",
        "white square sheet",
        "white square paper",
        "white square paper sheet",
        "white square card",
        "white paper square",
        "white sheet",
        "white paper",
        "square white sheet",
        "square white paper",
        "white square",
        "white card",
        "white piece of paper",
        "white paper sheet",
        "白色正方形纸片",
        "白色方形纸片",
        "白色正方形纸",
        "白色方形纸",
        "白色方纸片",
        "白色纸片",
        "白色纸",
        "白色卡片",
        "白色方形卡片",
        "白色正方形卡片",
        "白色薄片",
        "白色方形薄片",
        "白纸片",
        "白纸",
        "白卡片",
        "白色小纸片",
        "白色小方片",
        "白色方片",
        "白方片",
        "白色正方片",
        "白色方块纸",
    ),
    "black_rectangular_sheet": (
        "black_rectangular_sheet",
        "black rectangular sheet",
        "black rectangle sheet",
        "black rectangular paper",
        "black rectangle paper",
        "black rectangular paper sheet",
        "black rectangular card",
        "black rectangle card",
        "black paper rectangle",
        "black sheet",
        "black paper",
        "black card",
        "rectangular black sheet",
        "rectangular black paper",
        "black piece of paper",
        "black paper sheet",
        "black rectangle",
        "black rectangular piece",
        "黑色长方形纸片",
        "黑色矩形纸片",
        "黑色长方形纸",
        "黑色矩形纸",
        "黑色长纸片",
        "黑色纸片",
        "黑色纸",
        "黑色卡片",
        "黑色长方形卡片",
        "黑色矩形卡片",
        "黑色薄片",
        "黑色长方形薄片",
        "黑纸片",
        "黑纸",
        "黑卡片",
        "黑色小纸片",
        "黑色长方片",
        "黑色矩形片",
        "黑长方片",
        "黑矩形片",
        "黑色长片",
        "黑色方片",
    ),
    "red_cube": (
        "red_cube",
        "red cube",
        "red block",
        "red box",
        "red square block",
        "red cubic block",
        "red cubical block",
        "red object",
        "small red cube",
        "small red block",
        "the red cube",
        "the red block",
        "红色方块",
        "红方块",
        "红色立方体",
        "红立方体",
        "红色正方体",
        "红正方体",
        "红色块",
        "红块",
        "红色小方块",
        "红色小立方体",
        "红色小正方体",
        "小红块",
        "小红方块",
        "红色物体",
        "红色目标",
    ),
    "yellow_cylinder": (
        "yellow_cylinder",
        "yellow cylinder",
        "yellow cylindrical object",
        "yellow cylindrical block",
        "yellow round cylinder",
        "yellow round block",
        "yellow can",
        "yellow tube",
        "yellow object",
        "small yellow cylinder",
        "the yellow cylinder",
        "黄色圆柱",
        "黄圆柱",
        "黄色圆柱体",
        "黄圆柱体",
        "黄色柱体",
        "黄柱体",
        "黄色圆柱块",
        "黄色圆形柱体",
        "黄色圆筒",
        "黄圆筒",
        "黄色桶状物",
        "黄色小圆柱",
        "小黄圆柱",
        "黄色物体",
        "黄色目标",
    ),
    "cyan_cuboid": (
        "cyan_cuboid",
        "cyan cuboid",
        "cyan rectangular prism",
        "cyan rectangular block",
        "cyan block",
        "cyan box",
        "cyan object",
        "cyan long block",
        "cyan long cuboid",
        "teal cuboid",
        "teal rectangular prism",
        "teal rectangular block",
        "teal block",
        "turquoise cuboid",
        "turquoise rectangular prism",
        "turquoise rectangular block",
        "turquoise block",
        "blue green cuboid",
        "blue-green cuboid",
        "blue green block",
        "blue-green block",
        "small cyan cuboid",
        "the cyan cuboid",
        "青色长方体",
        "青长方体",
        "青色长方块",
        "青色方块",
        "青色块",
        "青色物体",
        "青色目标",
        "青色小长方体",
        "青色小方块",
        "蓝绿色长方体",
        "蓝绿色长方块",
        "蓝绿色方块",
        "蓝绿色块",
        "蓝绿色物体",
        "青绿色长方体",
        "青绿色长方块",
        "青绿色方块",
        "青绿色块",
        "青绿色物体",
        "湖蓝色长方体",
        "湖蓝色方块",
        "浅蓝绿色长方体",
        "浅蓝绿色方块",
        "蓝色长方体",
        "蓝色长方块",
        "蓝色方块",
    ),
}

POSITION_ALIASES = {
    "up": (
        "上方", "上侧", "上边", "上面", "上部", "上端", "上", "顶部",
        "顶端", "正上方", "up", "above", "top", "upper", "upper side",
        "top side", "on top", "on top of", "over", "overhead",
    ),
    "down": (
        "下方", "下侧", "下边", "下面", "下部", "下端", "下", "底部",
        "底端", "正下方", "down", "below", "bottom", "lower", "lower side",
        "bottom side", "under", "underneath", "beneath",
    ),
    "left": (
        "左方", "左侧", "左边", "左面", "左部", "左端", "左", "左手边",
        "左侧位置", "left", "left side", "to the left", "to the left of",
        "on the left", "on the left side", "left of",
    ),
    "right": (
        "右方", "右侧", "右边", "右面", "右部", "右端", "右", "右手边",
        "右侧位置", "right", "right side", "to the right", "to the right of",
        "on the right", "on the right side", "right of",
    ),
    "center": (
        "中心", "中央", "中间", "正中间", "正中心", "中部", "中", "中间位置",
        "中心位置", "center", "centre", "middle", "central", "in the center",
        "in the centre", "in the middle", "at the center", "at the centre",
        "middle position", "center position", "centre position",
    ),
}

MOVABLE_OBJECTS = frozenset({"red_cube", "yellow_cylinder", "cyan_cuboid"})


def _alias_pattern(aliases: dict[str, tuple[str, ...]]) -> str:
    values = [alias for group in aliases.values() for alias in group]
    values.sort(key=len, reverse=True)
    return "(?:" + "|".join(re.escape(value) for value in values) + ")"


def _reverse_aliases(aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {
        alias.casefold(): canonical
        for canonical, group in aliases.items()
        for alias in group
    }


OBJECT_BY_ALIAS = _reverse_aliases(OBJECT_ALIASES)
POSITION_BY_ALIAS = _reverse_aliases(POSITION_ALIASES)
OBJECT_PATTERN = _alias_pattern(OBJECT_ALIASES)
POSITION_PATTERN = _alias_pattern(POSITION_ALIASES)

CN_INSTRUCTION_PATTERN = re.compile(
    rf"(?:将|把)?\s*"
    rf"(?P<source>{OBJECT_PATTERN})\s*"
    rf"(?:移动|平移|放置|搬运|拿|拿起|放|放到|放在)\s*(?:到|至|在)?\s*"
    rf"(?P<target>{OBJECT_PATTERN})\s*(?:的)?\s*"
    rf"(?P<position>{POSITION_PATTERN})",
    flags=re.IGNORECASE,
)

EN_INSTRUCTION_PATTERN = re.compile(
    rf"(?:move|put|place|transfer|carry|pick\s+and\s+place)?\s*"
    rf"(?:the\s+)?(?P<source>{OBJECT_PATTERN})\s*"
    rf"(?:to|onto|on|at|in)?\s*"
    rf"(?:the\s+)?(?P<position>{POSITION_PATTERN})\s*"
    rf"(?:of|side\s+of)?\s*"
    rf"(?:the\s+)?(?P<target>{OBJECT_PATTERN})",
    flags=re.IGNORECASE,
)

INSTRUCTION_PATTERNS = (
    CN_INSTRUCTION_PATTERN,
    EN_INSTRUCTION_PATTERN,
)


@dataclass(frozen=True)
class PickPlaceCommand:
    source_object: str
    target_object: str
    target_position: str

    @property
    def source_site(self) -> str:
        return f"{self.source_object}_center_site"

    @property
    def destination_site(self) -> str:
        return f"{self.target_object}_{self.target_position}_site"

    @property
    def target_center_site(self) -> str:
        return f"{self.target_object}_center_site"


def parse_task(description: str) -> list[PickPlaceCommand]:
    commands = []
    for pattern in INSTRUCTION_PATTERNS:
        for match in pattern.finditer(description):
            source = OBJECT_BY_ALIAS[match.group("source").casefold()]
            target = OBJECT_BY_ALIAS[match.group("target").casefold()]
            position = POSITION_BY_ALIAS[match.group("position").casefold()]

            if source not in MOVABLE_OBJECTS:
                raise ValueError(f"object is not graspable: {source}")
            if source == target:
                raise ValueError(f"source and target objects must differ: {source}")

            command = PickPlaceCommand(
                source_object=source,
                target_object=target,
                target_position=position,
            )
            if command not in commands:
                commands.append(command)

    if not commands:
        raise ValueError(f"no pick-place instruction found in: {description!r}")
    return commands


class AutomatedTeleop:
    def __init__(
        self,
        env: LabSimMujocoEnv,
        motion_config: AutomatedTeleopConfig | None = None,
    ) -> None:
        self.env = env
        self.motion_config = motion_config or AutomatedTeleopConfig()

    def run(
        self,
        commands: list[PickPlaceCommand],
        *,
        viewer=None,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        callback_kwargs = (
            {} if step_callback is None else {"step_callback": step_callback}
        )
        self.env.set_gripper(
            False,
            settle_time_s=self.motion_config.gripper_settle_time_s,
            viewer=viewer,
            **callback_kwargs,
        )
        for index, command in enumerate(commands, start=1):
            print(
                f"[{index}/{len(commands)}] Move {command.source_object} "
                f"to {command.target_object} {command.target_position}."
            )
            self.execute_pick_place(
                command,
                viewer,
                step_callback=step_callback,
            )

    def execute_pick_place(
        self,
        command: PickPlaceCommand,
        viewer,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        callback_kwargs = (
            {} if step_callback is None else {"step_callback": step_callback}
        )
        pick_position = self.env.get_site_position(command.source_site)
        place_position = self.env.get_site_position(command.destination_site)
        target_center_position = self.env.get_site_position(command.target_center_site)
        place_position[2] += pick_position[2] - target_center_position[2]
        place_position[2] += self.motion_config.place_clearance_m
        hover_offset = np.asarray([0.0, 0.0, self.motion_config.hover_height_m])

        self.env.set_gripper(
            False,
            settle_time_s=self.motion_config.gripper_settle_time_s,
            viewer=viewer,
            **callback_kwargs,
        )
        self._move_to(pick_position + hover_offset, viewer, step_callback)
        self._move_to(pick_position, viewer, step_callback)
        self.env.hold(0.4, viewer=viewer, **callback_kwargs)
        self.env.set_gripper(
            True,
            settle_time_s=self.motion_config.gripper_settle_time_s,
            viewer=viewer,
            **callback_kwargs,
        )
        self._move_to(pick_position + hover_offset, viewer, step_callback)
        self._move_to(place_position + hover_offset, viewer, step_callback)
        self._move_to(place_position, viewer, step_callback)
        self.env.hold(0.4, viewer=viewer, **callback_kwargs)
        self.env.set_gripper(
            False,
            settle_time_s=self.motion_config.gripper_settle_time_s,
            viewer=viewer,
            **callback_kwargs,
        )
        self._move_to(place_position + hover_offset, viewer, step_callback)

    def _move_to(
        self,
        target_position: np.ndarray,
        viewer,
        step_callback: Callable[[], None] | None,
    ) -> None:
        callback_kwargs = (
            {} if step_callback is None else {"step_callback": step_callback}
        )
        self.env.move_pinch_to(
            target_position,
            max_speed_m_s=self.motion_config.move_speed_m_s,
            position_tolerance_m=self.motion_config.position_tolerance_m,
            max_motion_time_s=self.motion_config.max_motion_time_s,
            viewer=viewer,
            **callback_kwargs,
        )
