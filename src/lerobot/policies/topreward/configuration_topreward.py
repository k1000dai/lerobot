#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics.
Paper: https://arxiv.org/abs/2602.19313
Code: https://github.com/TOPReward/TOPReward
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig


@PreTrainedConfig.register_subclass("topreward")
@dataclass
class TOPRewardConfig(PreTrainedConfig):
    """Configuration for TOPReward zero-shot reward inference.

    TOPReward uses a video VLM's token probabilities as task-progress rewards. It is an
    inference-only reward model and does not predict robot actions.
    """

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_input_length: int = 32768
    fps: float = 2.0
    num_samples: int = 15
    reduction: str = "mean"
    true_token: str = "True"
    prompt_text: str = "The above video shows a robot manipulation trajectory that completes the following task: "
    add_chat_template: bool = False
    use_video_description: bool = False
    normalize_prefix_rewards: bool = True

    # Loading controls. Keep `attn_implementation` optional so macOS and CPU-only
    # environments do not require flash-attn.
    load_model: bool = True
    torch_dtype: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = True

    output_features: dict = field(
        default_factory=lambda: {
            "topreward": PolicyFeature(shape=(1,), type=FeatureType.REWARD),
        }
    )
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {self.reduction!r}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {self.num_samples}")
        if self.max_input_length < 1:
            raise ValueError(f"max_input_length must be at least 1, got {self.max_input_length}")

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return a placeholder optimizer config for config API compatibility."""
        return AdamWConfig()

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """TOPReward can score arbitrary frame lists, so dataset features are optional."""
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
