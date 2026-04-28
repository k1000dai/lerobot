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
import tomllib

import numpy as np
import pytest
import torch

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.topreward.configuration_topreward import TOPRewardConfig
from lerobot.policies.topreward.modeling_topreward import TOPRewardModel, TOPRewardOutput

# The LIBERO application script depends on the `dataset` and `pyarrow` extras —
# import lazily so this test module still loads in the base environment.
try:
    from lerobot.policies.topreward import compute_topreward_progress as ctp
    import pyarrow.parquet as pq

    _LIBERO_DEPS_AVAILABLE = True
except ImportError:
    ctp = None
    pq = None
    _LIBERO_DEPS_AVAILABLE = False

requires_libero_extras = pytest.mark.skipif(
    not _LIBERO_DEPS_AVAILABLE, reason="lerobot[dataset] + pyarrow required"
)


def test_topreward_extra_installs_accelerate_for_device_map_loading():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    topreward_extra = pyproject["project"]["optional-dependencies"]["topreward"]

    assert any(dependency.startswith("accelerate") for dependency in topreward_extra)


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


@requires_libero_extras
def test_to_uint8_image_handles_chw_float_tensor():
    chw = torch.zeros((3, 4, 5), dtype=torch.float32)
    chw[0, 0, 0] = 1.0  # red pixel @ (0,0)
    out = ctp._to_uint8_image(chw)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 255


@requires_libero_extras
def test_interpolate_to_frames_linear():
    out = ctp._interpolate_to_frames(np.array([1, 5]), np.array([0.0, 1.0]), 5)
    assert out.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


@requires_libero_extras
def test_compute_topreward_progress_writes_parquet(tmp_path, monkeypatch):
    """End-to-end glue test with a fake dataset and a fake VLM."""

    class FakeMeta:
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 4},
            {"dataset_from_index": 4, "dataset_to_index": 6},
        ]

    class FakeDataset:
        num_episodes = 2
        num_frames = 6

        def __init__(self):
            self.meta = FakeMeta()
            self.root = tmp_path

        def __getitem__(self, idx):
            return {
                "observation.images.image": np.zeros((4, 4, 3), dtype=np.uint8),
                "task": "open the drawer" if idx < 4 else "close the drawer",
            }

    captured = {"calls": []}

    def fake_compute(self, frames, instruction, **kwargs):  # noqa: ARG001
        captured["calls"].append((len(frames), instruction))
        return TOPRewardOutput(reward=float(len(frames)), reduction="mean", token_count=1)

    monkeypatch.setattr(ctp, "LeRobotDataset", lambda *a, **kw: FakeDataset())
    monkeypatch.setattr(ctp.TOPRewardModel, "compute_instruction_reward", fake_compute)
    monkeypatch.setattr(ctp.TOPRewardModel, "_load_qwen_backend", lambda self: None)

    out_path = tmp_path / "topreward_progress.parquet"
    ctp.compute_topreward_progress(
        dataset_repo_id="ignored/path",
        output_path=str(out_path),
        num_samples=3,
        device="cpu",
    )

    assert out_path.exists()
    table = pq.read_table(out_path)
    df = table.to_pandas()

    # 6 frames written across 2 episodes
    assert len(df) == 6
    assert sorted(df["episode_index"].unique().tolist()) == [0, 1]
    # normalize_prefix_rewards is True by default → values land in [0, 1]
    assert df["progress_topreward"].min() >= 0.0
    assert df["progress_topreward"].max() <= 1.0
    # First and last frame of each episode hit the prefix endpoints
    ep0 = df[df["episode_index"] == 0].sort_values("frame_index")
    assert ep0["progress_topreward"].iloc[0] == pytest.approx(0.0)
    assert ep0["progress_topreward"].iloc[-1] == pytest.approx(1.0)
    # Each task description was forwarded to the VLM
    instructions = {call[1] for call in captured["calls"]}
    assert instructions == {"open the drawer", "close the drawer"}


@requires_libero_extras
def test_visualize_topreward_episode_writes_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    frames = [np.full((6, 6, 3), i * 32, dtype=np.uint8) for i in range(8)]
    prefix_lengths = np.array([1, 4, 8])
    prefix_rewards = np.array([0.0, 0.5, 1.0])
    per_frame = ctp._interpolate_to_frames(prefix_lengths, prefix_rewards, len(frames))

    out_path = tmp_path / "ep.png"
    ctp.visualize_topreward_episode(
        frames=frames,
        prefix_lengths=prefix_lengths,
        prefix_rewards=prefix_rewards,
        per_frame_rewards=per_frame,
        task="pick up the block",
        output_path=out_path,
        num_thumbnails=4,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


@requires_libero_extras
def test_compute_topreward_progress_emits_visualizations(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    class FakeMeta:
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 4},
            {"dataset_from_index": 4, "dataset_to_index": 6},
        ]

    class FakeDataset:
        num_episodes = 2
        num_frames = 6

        def __init__(self):
            self.meta = FakeMeta()
            self.root = tmp_path

        def __getitem__(self, idx):
            return {
                "observation.images.image": np.full((4, 4, 3), 128, dtype=np.uint8),
                "task": "task A" if idx < 4 else "task B",
            }

    def fake_compute(self, frames, instruction, **kwargs):  # noqa: ARG001
        return TOPRewardOutput(reward=float(len(frames)), reduction="mean", token_count=1)

    monkeypatch.setattr(ctp, "LeRobotDataset", lambda *a, **kw: FakeDataset())
    monkeypatch.setattr(ctp.TOPRewardModel, "compute_instruction_reward", fake_compute)
    monkeypatch.setattr(ctp.TOPRewardModel, "_load_qwen_backend", lambda self: None)

    viz_dir = tmp_path / "plots"
    ctp.compute_topreward_progress(
        dataset_repo_id="ignored/path",
        output_path=str(tmp_path / "topreward_progress.parquet"),
        num_samples=3,
        device="cpu",
        num_visualizations=1,
        viz_dir=str(viz_dir),
    )

    pngs = sorted(viz_dir.glob("*.png"))
    # Only the first processed episode should be plotted
    assert len(pngs) == 1
    assert pngs[0].name == "topreward_ep0000.png"
