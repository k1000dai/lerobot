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

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_topreward import TOPRewardConfig


@dataclass
class TOPRewardOutput:
    """Output metadata for a TOPReward log-likelihood computation."""

    reward: float
    reduction: str
    token_count: int
    per_token_log_probs: list[float] | None = None
    token_ids: list[int] | None = None
    prefix_lengths: list[int] | None = None
    prefix_rewards: list[float] | None = None
    normalized_prefix_rewards: list[float] | None = None
    trajectory_description: str | None = None


class TOPRewardModel(PreTrainedPolicy):
    """Zero-shot reward model based on VLM token probabilities.

    This implementation follows TOPReward's core inference path: ask a video VLM
    whether a trajectory completes an instruction, then use the log-probability of
    the affirmative token as the reward.
    """

    name = "topreward"
    config_class = TOPRewardConfig

    def __init__(
        self,
        config: TOPRewardConfig,
        *,
        model: nn.Module | None = None,
        processor: Any | None = None,
        process_vision_info: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model
        self.processor = processor
        self._process_vision_info = process_vision_info

        if config.load_model and (self.model is None or self.processor is None or self._process_vision_info is None):
            self._load_qwen_backend()

        if self.model is not None and config.device_map is None:
            self.model.to(self.device)

    def _load_qwen_backend(self) -> None:
        require_package("transformers", extra="topreward")
        require_package("qwen-vl-utils", extra="topreward", import_name="qwen_vl_utils")

        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "TOPReward requires a transformers version with Qwen3VLForConditionalGeneration. "
                "Install the 'topreward' extra or update transformers."
            ) from e

        model_kwargs: dict[str, Any] = {"torch_dtype": self.config.torch_dtype}
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.config.model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=self.config.trust_remote_code
        )
        self._process_vision_info = process_vision_info

    @staticmethod
    def _to_pil(frame: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu()
            if frame.ndim == 3 and frame.shape[0] in {1, 3}:
                frame = frame.permute(1, 2, 0)
            frame = frame.numpy()

        array = np.asarray(frame)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array)

    @staticmethod
    def normalize_rewards(rewards: Sequence[float]) -> np.ndarray:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.size == 0:
            return rewards_arr
        if rewards_arr.size == 1:
            return np.ones_like(rewards_arr)

        min_reward = rewards_arr.min()
        max_reward = rewards_arr.max()
        if max_reward == min_reward:
            return np.ones_like(rewards_arr)
        return (rewards_arr - min_reward) / (max_reward - min_reward)

    def _ensure_backend_loaded(self) -> None:
        if self.model is None or self.processor is None or self._process_vision_info is None:
            raise RuntimeError(
                "TOPReward backend is not loaded. Set load_model=True or pass model, processor, "
                "and process_vision_info when constructing TOPRewardModel."
            )

    def _target_token_count(self, token: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return 1
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        return max(1, len(token_ids))

    def generate_trajectory_description(
        self,
        frames: Sequence[Image.Image | np.ndarray | torch.Tensor],
        *,
        fps: float | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an instruction-agnostic trajectory description with the VLM."""
        self._ensure_backend_loaded()
        pil_frames = [self._to_pil(frame) for frame in frames]
        content = [
            {"type": "video", "video": pil_frames, "fps": fps or self.config.fps},
            {"type": "text", "text": "Describe the robot manipulation trajectory in this video:"},
        ]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        prompt_text = self.processor.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response[len(prompt_text) :].strip() if response.startswith(prompt_text) else response.strip()

    @torch.no_grad()
    def compute_instruction_reward(
        self,
        frames: Sequence[Image.Image | np.ndarray | torch.Tensor],
        instruction: str,
        *,
        reduction: str | None = None,
        fps: float | None = None,
        use_video_description: bool | None = None,
        add_chat_template: bool | None = None,
    ) -> TOPRewardOutput:
        """Compute the TOPReward log-likelihood for a trajectory and instruction."""
        self._ensure_backend_loaded()
        if len(frames) == 0:
            raise ValueError("frames must contain at least one frame")

        reduction = reduction or self.config.reduction
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

        fps = fps or self.config.fps
        use_video_description = self.config.use_video_description if use_video_description is None else use_video_description
        add_chat_template = self.config.add_chat_template if add_chat_template is None else add_chat_template

        pil_frames = [self._to_pil(frame) for frame in frames]
        trajectory_description = None
        if use_video_description:
            trajectory_description = self.generate_trajectory_description(pil_frames, fps=fps)
            prompt_text = (
                f"{trajectory_description} Therefore given the above description and the video, "
                "the video shows a robot manipulation trajectory that **completes** the following instruction: "
            )
        else:
            prompt_text = self.config.prompt_text

        content = [
            {"type": "video", "video": pil_frames, "fps": fps},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
        eos_token = getattr(getattr(self.processor, "tokenizer", None), "eos_token", None)

        if add_chat_template:
            instruction_suffix = f"{instruction} Decide whether the above statement is True or not.\nThe answer is:"
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": fps},
                        {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = f"{prompt_chat}{self.config.true_token}"
            vision_messages = templated_messages
        else:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True or not.\n"
                f"The answer is: {self.config.true_token}"
            )
            prompt_chat = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            vision_messages = messages

        image_inputs, video_inputs = self._process_vision_info(vision_messages)
        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.config.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds maximum of {self.config.max_input_length} tokens")

        labels = inputs["input_ids"].clone()
        target_token_count = self._target_token_count(self.config.true_token)
        prompt_length = max(0, labels.shape[1] - target_token_count)
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        outputs = self.model(**inputs, labels=labels)
        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = functional.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        if not mask.any():
            raise ValueError("No target tokens were available for TOPReward scoring")

        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]
        reward = masked_log_probs.sum().item() if reduction == "sum" else masked_log_probs.mean().item()

        return TOPRewardOutput(
            reward=reward,
            reduction=reduction,
            token_count=int(masked_log_probs.numel()),
            per_token_log_probs=masked_log_probs.detach().cpu().tolist(),
            token_ids=target_labels[mask].detach().cpu().tolist(),
            trajectory_description=trajectory_description,
        )

    def calculate_rewards(
        self,
        frames: Sequence[Image.Image | np.ndarray | torch.Tensor],
        instruction: str,
        *,
        num_samples: int | None = None,
        return_all_frames: bool = False,
        reduction: str | None = None,
        fps: float | None = None,
        use_video_description: bool | None = None,
        add_chat_template: bool | None = None,
    ) -> float | np.ndarray:
        """Calculate TOPReward progress values for a full trajectory or sampled prefixes."""
        if len(frames) == 0:
            raise ValueError("frames must contain at least one frame")

        if not return_all_frames:
            return self.compute_instruction_reward(
                frames,
                instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=use_video_description,
                add_chat_template=add_chat_template,
            ).reward

        num_samples = min(num_samples or self.config.num_samples, len(frames))
        prefix_lengths = np.linspace(1, len(frames), num_samples, dtype=int)
        prefix_lengths = sorted({int(length) for length in prefix_lengths})

        rewards = [
            self.compute_instruction_reward(
                frames[:length],
                instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=use_video_description,
                add_chat_template=add_chat_template,
            ).reward
            for length in prefix_lengths
        ]
        if self.config.normalize_prefix_rewards:
            return self.normalize_rewards(rewards)
        return np.asarray(rewards, dtype=np.float64)

    def get_optim_params(self):
        """TOPReward is inference-only in LeRobot; there are no trainable params to optimize."""
        return []

    def reset(self):
        return None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("TOPReward is a zero-shot inference reward model and does not train via forward")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("TOPReward is a reward model and does not predict action chunks")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("TOPReward is a reward model and does not select actions")
