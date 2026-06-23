from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = (
    REPO_ROOT / "thirdparty" / "openpi" / "packages" / "openpi-client" / "src"
)

for import_path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from dataset_record.config import EXTERIOR_IMAGE_KEY, WRIST_IMAGE_KEY, RecordConfig
from env import LabSimMujocoEnv, StateSample
from openpi_client import image_tools, websocket_client_policy


STATE_KEY = "observation.state"
PROMPT_KEY = "prompt"
ACTION_KEY = "actions"
ACTION_DIM = 7

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000


def build_ur5e_observation(
    state_sample: StateSample,
    *,
    prompt: str,
    image_size: int = 224,
) -> dict[str, Any]:
    """Build the observation dict expected by openpi.policies.ur5e_policy.UR5Inputs."""
    missing_images = [
        key
        for key in (EXTERIOR_IMAGE_KEY, WRIST_IMAGE_KEY)
        if key not in state_sample.images
    ]
    if missing_images:
        raise KeyError(f"state sample is missing image key(s): {missing_images}")

    return {
        STATE_KEY: _as_state(state_sample.state),
        EXTERIOR_IMAGE_KEY: _as_image(
            state_sample.images[EXTERIOR_IMAGE_KEY],
            image_size=image_size,
        ),
        WRIST_IMAGE_KEY: _as_image(
            state_sample.images[WRIST_IMAGE_KEY],
            image_size=image_size,
        ),
        PROMPT_KEY: prompt,
    }


def capture_ur5e_observation(
    sim_env: LabSimMujocoEnv,
    *,
    prompt: str,
    gripper_closed: bool | None = None,
    image_size: int = 224,
) -> dict[str, Any]:
    """Capture state/images from LabSimMujocoEnv and format them for the server."""
    if gripper_closed is None:
        gripper_closed = _infer_gripper_closed(sim_env)
    state_sample = sim_env.capture_state(gripper_closed=gripper_closed)
    return build_ur5e_observation(
        state_sample,
        prompt=prompt,
        image_size=image_size,
    )


class RemoteUR5EInferenceClient:
    """Small wrapper around openpi-client for the lab_sim UR5E observation format."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int | None = DEFAULT_PORT,
        api_key: str | None = None,
        image_size: int = 224,
    ) -> None:
        self.image_size = image_size
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )

    def infer_action_chunk_from_env(
        self,
        sim_env: LabSimMujocoEnv,
        *,
        prompt: str,
        gripper_closed: bool | None = None,
    ) -> np.ndarray:
        observation = capture_ur5e_observation(
            sim_env,
            prompt=prompt,
            gripper_closed=gripper_closed,
            image_size=self.image_size,
        )
        return self.infer_action_chunk(observation)

    def infer_action_chunk_from_sample(
        self,
        state_sample: StateSample,
        *,
        prompt: str,
    ) -> np.ndarray:
        observation = build_ur5e_observation(
            state_sample,
            prompt=prompt,
            image_size=self.image_size,
        )
        return self.infer_action_chunk(observation)

    def infer_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        result = self._policy.infer(observation)
        if ACTION_KEY not in result:
            raise KeyError(
                f"policy response did not contain {ACTION_KEY!r}; got {list(result)}"
            )

        action_chunk = np.asarray(result[ACTION_KEY], dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[-1] != ACTION_DIM:
            raise ValueError(
                "expected action chunk with shape "
                f"(action_horizon, {ACTION_DIM}), got {action_chunk.shape}"
            )
        return action_chunk

    def reset(self) -> None:
        self._policy.reset()


def _as_state(state: Any) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (ACTION_DIM,):
        raise ValueError(f"expected {STATE_KEY} shape {(ACTION_DIM,)}, got {state_array.shape}")
    return np.ascontiguousarray(state_array)


def _as_image(image: Any, *, image_size: int) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(f"expected HWC image with 3 dims, got {image_array.shape}")

    if image_array.shape[0] == 3 and image_array.shape[-1] != 3:
        image_array = np.moveaxis(image_array, 0, -1)
    if image_array.shape[-1] != 3:
        raise ValueError(f"expected RGB image with 3 channels, got {image_array.shape}")

    image_array = image_tools.convert_to_uint8(image_array)
    image_array = image_tools.resize_with_pad(image_array, image_size, image_size)
    return np.ascontiguousarray(image_array)


def _infer_gripper_closed(sim_env: LabSimMujocoEnv) -> bool:
    return sim_env.gripper_closed


def parse_args() -> argparse.Namespace:
    cfg = RecordConfig()
    parser = argparse.ArgumentParser(
        description="Query a remote openpi UR5E policy server from LabSimMujocoEnv."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default=cfg.task)
    parser.add_argument("--fps", type=int, default=cfg.fps)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print the full returned action chunk instead of only its first row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        RecordConfig(),
        task=args.prompt,
        fps=args.fps,
        image_size=args.image_size,
    )

    client = RemoteUR5EInferenceClient(
        host=args.host,
        port=args.port,
        image_size=args.image_size,
    )
    sim_env = LabSimMujocoEnv(cfg)
    try:
        sim_env.reset()
        action_chunk = client.infer_action_chunk_from_env(
            sim_env,
            prompt=args.prompt,
        )
        print(f"action_chunk shape={tuple(action_chunk.shape)} dtype={action_chunk.dtype}")
        if args.print_actions:
            np.set_printoptions(precision=5, suppress=True)
            print(action_chunk)
        else:
            print(f"first_action={action_chunk[0].tolist()}")
    finally:
        sim_env.close()


if __name__ == "__main__":
    main()
