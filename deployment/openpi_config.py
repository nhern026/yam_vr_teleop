"""YAM configuration for XPolicyLab/OpenPI; import in the training environment.

Construct with make_config(repo_id='local/yam', exp_name='place-vial'). The
standard scripts can receive this config through scripts/openpi_yam.py.
"""
import dataclasses
from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config, weight_loaders
from deployment.yam_policy import YamInputs, YamOutputs


@dataclasses.dataclass(frozen=True)
class YamDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                "state": "observation.state",
                "images/left": "observation.images.cam_left_wrist",
                "images/right": "observation.images.cam_right_wrist",
                "images/base": "observation.images.cam_overhead",
                "actions": "action", "prompt": "prompt",
            })]),
            data_transforms=transforms.Group(inputs=[YamInputs()], outputs=[YamOutputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def make_config(repo_id, exp_name="yam", batch_size=8):
    return config.TrainConfig(
        name="pi05_yam", exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=YamDataConfig(repo_id=repo_id, base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=batch_size, fsdp_devices=1, num_train_steps=30000,
    )
