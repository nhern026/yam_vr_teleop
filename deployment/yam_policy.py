"""Three-view OpenPI input transform; use instead of ALOHA's joint conversion.

Repack dataset keys into state, images/left, images/right, images/base, actions and prompt
before this transform. Append OpenPI's usual resize/normalization/model
transforms afterwards. Older two-camera data gets a masked-out base view.
"""
import numpy as np

class YamInputs:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError("YAM state must be 14-dimensional")
        images = {}
        for side in ("left", "right"):
            image = np.asarray(data[f"images/{side}"])
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.transpose(1, 2, 0)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("expected RGB image")
            if np.issubdtype(image.dtype, np.floating):
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            images[side] = image
        base = data.get("images/base")
        if base is not None:
            base = np.asarray(base)
            if base.ndim == 3 and base.shape[0] == 3 and base.shape[-1] != 3:
                base = base.transpose(1, 2, 0)
            if base.ndim != 3 or base.shape[-1] != 3:
                raise ValueError("expected RGB base image")
            if np.issubdtype(base.dtype, np.floating):
                base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        else:
            base = np.zeros_like(images["left"])
        result = {"state": state, "image": {"base_0_rgb": base,
                   "left_wrist_0_rgb": images["left"], "right_wrist_0_rgb": images["right"]},
                  "image_mask": {"base_0_rgb": np.bool_("images/base" in data), "left_wrist_0_rgb": np.bool_(True),
                                 "right_wrist_0_rgb": np.bool_(True)}}
        for key in ("actions", "prompt"):
            if key in data:
                result[key] = data[key]
        return result

class YamOutputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :14]}
