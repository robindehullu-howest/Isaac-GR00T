# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
    else:
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    
    try:
        from os import environ
        from gr00t.data.gcs_utils import push_models_to_gcs
        bucket_name = environ.get("MODELS_BUCKET_NAME", "robot-445714_lerobot_models")
        base_dir = Path(output_dir).parents[1].resolve()
        model_id = str(Path(output_dir).relative_to(base_dir))
        push_models_to_gcs(bucket_name=bucket_name, base_dir=base_dir, model_ids=[model_id])
    except Exception as e:
        print(f"Error pushing model to GCS: {e}")
        pass


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            output_dir = Path(args.output_dir)
            checkpoint_dir = Path(output_dir / f"checkpoint-{state.global_step}").resolve()

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            try:
                from os import environ
                from gr00t.data.gcs_utils import push_models_to_gcs
                bucket_name = environ.get("MODELS_BUCKET_NAME", "robot-445714_lerobot_models")
                base_dir = output_dir.parents[1].resolve()
                model_id = str(checkpoint_dir.relative_to(base_dir))
                push_models_to_gcs(bucket_name=bucket_name, base_dir=base_dir, model_ids=[model_id])
            except Exception as e:
                print(f"Error pushing model to GCS: {e}")
                pass
