from __future__ import annotations

import json
import random
from itertools import permutations, product
from pathlib import Path


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "info" / "task1"
OUTPUT_FILE = OUTPUT_DIRECTORY / "task_descriptions.json"
TASK_DESCRIPTION_COUNT = 200
SELECTION_SEED = 20260621

MOVABLE_OBJECTS = ("red_cube", "yellow_cylinder", "cyan_cuboid")
SHEET_OBJECTS = ("white_square_sheet", "black_rectangular_sheet")
ALL_OBJECTS = (
    "red_cube",
    "yellow_cylinder",
    "cyan_cuboid",
    "white_square_sheet",
    "black_rectangular_sheet",
)
POSITIONS = ("up", "down", "left", "right", "center")
NON_CENTER_POSITIONS = ("up", "down", "left", "right")

OBJECT_NAMES = {
    "red_cube": {"zh": "红色方块", "en": "red cube"},
    "yellow_cylinder": {"zh": "黄色圆柱", "en": "yellow cylinder"},
    "cyan_cuboid": {"zh": "青色长方体", "en": "cyan cuboid"},
    "white_square_sheet": {"zh": "白色正方形纸片", "en": "white square paper"},
    "black_rectangular_sheet": {
        "zh": "黑色长方形纸片",
        "en": "black rectangular paper",
    },
}

POSITION_NAMES = {
    "up": {
        "zh": ("上方", "上侧", "正上方"),
        "en": ("above", "on top of", "over"),
    },
    "down": {
        "zh": ("下方", "下侧", "正下方"),
        "en": ("below", "under", "beneath"),
    },
    "left": {
        "zh": ("左侧", "左边", "左手边"),
        "en": ("to the left of", "on the left side of", "left of"),
    },
    "right": {
        "zh": ("右侧", "右边", "右手边"),
        "en": ("to the right of", "on the right side of", "right of"),
    },
    "center": {
        "zh": ("中心", "中央", "正中间"),
        "en": ("at the center of", "in the centre of", "in the middle of"),
    },
}

CHINESE_TEMPLATES = (
    "将{a}放置到{b}的{ab_position}，然后将{c}放置到{d}的{cd_position}。",
    "把{a}移动到{b}的{ab_position}，再把{c}放到{d}的{cd_position}。",
    "将{a}平移到{b}的{ab_position}，然后将{c}放置在{d}的{cd_position}。",
    "把{a}搬运到{b}的{ab_position}，接着把{c}移动到{d}的{cd_position}。",
)

ENGLISH_TEMPLATES = (
    "Place the {a} {ab_position} the {b}, then place the {c} {cd_position} the {d}.",
    "Move the {a} {ab_position} the {b}, then put the {c} {cd_position} the {d}.",
    "Transfer the {a} {ab_position} the {b}, then carry the {c} {cd_position} the {d}.",
    "Pick and place the {a} {ab_position} the {b}. Then move the {c} {cd_position} the {d}.",
)

Command = tuple[str, str, str]
TaskScenario = tuple[Command, Command]


def _positions_for_target(target_object: str) -> tuple[str, ...]:
    if target_object in SHEET_OBJECTS:
        return POSITIONS
    return NON_CENTER_POSITIONS


def _all_task1_scenarios() -> list[TaskScenario]:
    scenarios = []
    for source_a, source_c in permutations(MOVABLE_OBJECTS, 2):
        target_candidates = tuple(
            object_name
            for object_name in ALL_OBJECTS
            if object_name not in {source_a, source_c}
        )
        for target_b, target_d in permutations(target_candidates, 2):
            position_pairs = product(
                _positions_for_target(target_b),
                _positions_for_target(target_d),
            )
            for position_ab, position_cd in position_pairs:
                scenarios.append(
                    (
                        (source_a, target_b, position_ab),
                        (source_c, target_d, position_cd),
                    )
                )
    return scenarios


def _coverage_features(scenario: TaskScenario) -> frozenset[tuple[str, ...]]:
    (source_a, target_b, position_ab), (source_c, target_d, position_cd) = scenario
    return frozenset(
        {
            ("first_source", source_a),
            ("second_source", source_c),
            ("source_pair", source_a, source_c),
            ("first_target", target_b),
            ("second_target", target_d),
            ("target_pair", target_b, target_d),
            ("first_position", position_ab),
            ("second_position", position_cd),
            ("position_pair", position_ab, position_cd),
            ("first_source_target", source_a, target_b),
            ("second_source_target", source_c, target_d),
            ("first_source_position", source_a, position_ab),
            ("second_source_position", source_c, position_cd),
            ("first_target_position", target_b, position_ab),
            ("second_target_position", target_d, position_cd),
        }
    )


def _select_for_coverage(
    scenarios: list[TaskScenario],
    count: int,
) -> list[TaskScenario]:
    if count > len(scenarios):
        raise ValueError(f"requested {count} descriptions from {len(scenarios)} scenarios")

    candidates = scenarios.copy()
    random.Random(SELECTION_SEED).shuffle(candidates)
    selected = []
    covered_features: set[tuple[str, ...]] = set()

    while len(selected) < count:
        best_index = max(
            range(len(candidates)),
            key=lambda index: len(
                _coverage_features(candidates[index]) - covered_features
            ),
        )
        scenario = candidates.pop(best_index)
        selected.append(scenario)
        covered_features.update(_coverage_features(scenario))

    return selected


def _render_task1_description(
    scenario: TaskScenario,
    index: int,
    position_variant_indices: tuple[int, int],
) -> dict[str, object]:
    (source_a, target_b, position_ab), (source_c, target_d, position_cd) = scenario
    template_index = index % len(CHINESE_TEMPLATES)
    position_ab_variant, position_cd_variant = position_variant_indices

    chinese_values = {
        "a": OBJECT_NAMES[source_a]["zh"],
        "b": OBJECT_NAMES[target_b]["zh"],
        "c": OBJECT_NAMES[source_c]["zh"],
        "d": OBJECT_NAMES[target_d]["zh"],
        "ab_position": POSITION_NAMES[position_ab]["zh"][position_ab_variant],
        "cd_position": POSITION_NAMES[position_cd]["zh"][position_cd_variant],
    }
    english_values = {
        "a": OBJECT_NAMES[source_a]["en"],
        "b": OBJECT_NAMES[target_b]["en"],
        "c": OBJECT_NAMES[source_c]["en"],
        "d": OBJECT_NAMES[target_d]["en"],
        "ab_position": POSITION_NAMES[position_ab]["en"][position_ab_variant],
        "cd_position": POSITION_NAMES[position_cd]["en"][position_cd_variant],
    }

    return {
        "id": f"task1_{index + 1:03d}",
        "chinese": CHINESE_TEMPLATES[template_index].format(**chinese_values),
        "english": ENGLISH_TEMPLATES[template_index].format(**english_values),
        "commands": [
            {
                "source_object": source_a,
                "target_object": target_b,
                "target_position": position_ab,
            },
            {
                "source_object": source_c,
                "target_object": target_d,
                "target_position": position_cd,
            },
        ],
    }


def generate_task1_descriptions() -> list[dict[str, object]]:
    scenarios = _select_for_coverage(
        _all_task1_scenarios(),
        TASK_DESCRIPTION_COUNT,
    )
    position_usage = {position: 0 for position in POSITIONS}
    descriptions = []
    for index, scenario in enumerate(scenarios):
        position_ab = scenario[0][2]
        position_cd = scenario[1][2]
        position_ab_variant = position_usage[position_ab] % 3
        position_usage[position_ab] += 1
        position_cd_variant = position_usage[position_cd] % 3
        position_usage[position_cd] += 1
        descriptions.append(
            _render_task1_description(
                scenario,
                index,
                (position_ab_variant, position_cd_variant),
            )
        )
    return descriptions


def generate_task2_descriptions() -> list[dict[str, object]]:
    raise NotImplementedError("task2 description generation is not implemented")


def generate_task3_descriptions() -> list[dict[str, object]]:
    raise NotImplementedError("task3 description generation is not implemented")


def main() -> None:
    descriptions = generate_task1_descriptions()
    payload = {
        "task": "task1",
        "count": len(descriptions),
        "descriptions": descriptions,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(descriptions)} task descriptions: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
