import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


STATE_KEY = "observation.state"
EXTERIOR_IMAGE_KEY = "observation.images.exterior_image_1_left"
WRIST_IMAGE_KEY = "observation.images.wrist_image_left"
ACTION_DIM = 7


def make_ur5e_example() -> dict:
    """Create an example observation in the format used by the lab simulator."""
    return {
        STATE_KEY: np.random.rand(ACTION_DIM).astype(np.float32),
        EXTERIOR_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        WRIST_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    # LeRobot decodes video frames as float32 CHW arrays in [0, 1].
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI05:
            raise ValueError(f"UR5Inputs only supports PI0.5, got {self.model_type}")

        state = np.asarray(data[STATE_KEY], dtype=np.float32)
        base_image = _parse_image(data[EXTERIOR_IMAGE_KEY])
        wrist_image = _parse_image(data[WRIST_IMAGE_KEY])
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
