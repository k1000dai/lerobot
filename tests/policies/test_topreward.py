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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.topreward.configuration_topreward import TOPRewardConfig
from lerobot.policies.topreward.modeling_topreward import TOPRewardModel, TOPRewardOutput


class FakeInputs(dict):
    def to(self, device):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token="<eos>")
        self.messages = None
        self.text = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.messages = messages
        return "chat-prefix "

    def __call__(self, *, text, images=None, videos=None, padding=True, return_tensors="pt"):
        self.text = text[0]
        return FakeInputs(
            {
                "input_ids": torch.tensor([[1, 2, 7]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )


class FakeQwenModel(torch.nn.Module):
    def forward(self, **kwargs):
        logits = torch.zeros((1, 3, 10), dtype=torch.float32)
        logits[0, 1, 7] = 3.0
        return SimpleNamespace(logits=logits)


def test_topreward_factory_registration():
    cfg = make_policy_config("topreward", device="cpu", load_model=False)

    assert isinstance(cfg, TOPRewardConfig)
    assert get_policy_class("topreward") is TOPRewardModel
    assert cfg.type == "topreward"
    assert cfg.model_name == "Qwen/Qwen3-VL-8B-Instruct"


def test_compute_instruction_reward_uses_true_token_log_probability():
    processor = FakeProcessor()
    model = TOPRewardModel(
        TOPRewardConfig(device="cpu", load_model=False),
        model=FakeQwenModel(),
        processor=processor,
        process_vision_info=lambda messages: (None, None),
    )

    output = model.compute_instruction_reward(
        frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        instruction="open the drawer",
    )

    expected = torch.log_softmax(torch.tensor([0.0] * 7 + [3.0] + [0.0] * 2), dim=-1)[7].item()
    assert isinstance(output, TOPRewardOutput)
    assert output.reward == pytest.approx(expected)
    assert output.token_count == 1
    assert output.token_ids == [7]
    assert "open the drawer" in processor.text
    assert "True" in processor.text


def test_calculate_rewards_for_prefixes_normalizes_progress(monkeypatch):
    model = TOPRewardModel(TOPRewardConfig(device="cpu", load_model=False))

    def fake_compute_instruction_reward(frames, instruction, **kwargs):
        return TOPRewardOutput(reward=float(len(frames)), reduction="mean", token_count=1)

    monkeypatch.setattr(model, "compute_instruction_reward", fake_compute_instruction_reward)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)]

    rewards = model.calculate_rewards(
        frames=frames,
        instruction="stack the blocks",
        num_samples=3,
        return_all_frames=True,
    )

    assert rewards.tolist() == [0.0, 0.5, 1.0]
