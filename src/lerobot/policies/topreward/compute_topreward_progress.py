#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Apply TOPReward to a LIBERO LeRobot dataset.

For each episode the script:
1. Loads the agentview frames and the natural-language task description.
2. Asks ``TOPRewardModel.calculate_rewards`` to score ``num_samples`` evenly
   spaced prefixes (this is what TOPReward calls "instruction-conditioned
   progress"; the prefixes are normalized to ``[0, 1]`` when
   ``normalize_prefix_rewards`` is set, which is the paper's default).
3. Linearly interpolates the prefix progress back to a per-frame value so the
   output schema matches ``compute_rabc_weights.py`` (one row per frame), and
   downstream RA-BC weighting can consume both reward signals interchangeably.

Default image key (``observation.images.image``) follows the LIBERO env's
``LIBERO_KEY_PIXELS_AGENTVIEW`` mapping in ``src/lerobot/envs/configs.py``.

Example:
    uv run python -m lerobot.policies.topreward.compute_topreward_progress \\
        --dataset-repo-id HuggingFaceVLA/libero \\
        --num-samples 15 \\
        --episode-indices 0,1,2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

from .configuration_topreward import TOPRewardConfig
from .modeling_topreward import TOPRewardModel


def _to_uint8_image(img: Any) -> np.ndarray:
    """Convert a LeRobotDataset image sample to ``(H, W, 3)`` uint8 for the VLM."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    array = np.asarray(img)
    if array.ndim == 4:
        array = array[array.shape[0] // 2]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def _load_episode_frames(
    dataset: LeRobotDataset, ep_start: int, ep_end: int, image_key: str
) -> tuple[list[np.ndarray], str]:
    """Return all decoded frames for a single episode plus its task description."""
    frames: list[np.ndarray] = []
    task: str = "perform the task"
    for frame_idx in range(ep_start, ep_end):
        sample = dataset[frame_idx]
        if image_key not in sample:
            raise KeyError(
                f"Image key {image_key!r} not in dataset sample. Available keys: {sorted(sample.keys())}"
            )
        frames.append(_to_uint8_image(sample[image_key]))
        if frame_idx == ep_start:
            task = sample.get("task", task)
    return frames, str(task)


def _interpolate_to_frames(
    prefix_lengths: np.ndarray, prefix_rewards: np.ndarray, num_frames: int
) -> np.ndarray:
    """Interpolate sparse prefix-level rewards to one value per episode frame."""
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(prefix_lengths) == 1:
        return np.full(num_frames, float(prefix_rewards[0]), dtype=np.float32)
    # ``prefix_lengths`` are 1-indexed (prefix length L covers frames [0, L-1]),
    # so the value applies to local frame index ``L-1``.
    xs = np.asarray(prefix_lengths, dtype=np.float64) - 1.0
    ys = np.asarray(prefix_rewards, dtype=np.float64)
    targets = np.arange(num_frames, dtype=np.float64)
    return np.interp(targets, xs, ys).astype(np.float32)


def visualize_topreward_episode(
    *,
    frames: list[np.ndarray],
    prefix_lengths: np.ndarray,
    prefix_rewards: np.ndarray,
    per_frame_rewards: np.ndarray,
    task: str,
    output_path: Path,
    num_thumbnails: int = 8,
) -> None:
    """Plot per-frame TOPReward progress + sampled thumbnails for one episode."""
    import matplotlib.pyplot as plt  # imported lazily — matplotlib is optional

    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("Cannot visualize an empty episode")
    if len(per_frame_rewards) != num_frames:
        raise ValueError(
            f"per_frame_rewards length {len(per_frame_rewards)} does not match num_frames {num_frames}"
        )

    frame_indices = np.arange(num_frames)
    normalized = float(per_frame_rewards.min()) >= -1e-6 and float(per_frame_rewards.max()) <= 1.0 + 1e-6
    y_label = "Progress (normalized)" if normalized else "Log-prob reward"

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax_progress = fig.add_subplot(gs[0])
    ax_frames = fig.add_subplot(gs[1])

    ax_progress.plot(
        frame_indices,
        per_frame_rewards,
        color="#2E86AB",
        linewidth=2,
        label="TOPReward (interpolated)",
    )
    ax_progress.fill_between(frame_indices, 0, per_frame_rewards, alpha=0.25, color="#2E86AB")
    ax_progress.scatter(
        np.asarray(prefix_lengths) - 1,
        prefix_rewards,
        color="#E63946",
        zorder=5,
        s=40,
        label=f"Prefix samples ({len(prefix_lengths)})",
    )
    ax_progress.set_xlabel("Frame")
    ax_progress.set_ylabel(y_label)
    ax_progress.set_title(f'Task: "{task}"', fontweight="bold")
    if normalized:
        ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    ax_frames.axis("off")
    n_thumbs = min(num_thumbnails, num_frames)
    sample_indices = np.linspace(0, num_frames - 1, n_thumbs, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * n_thumbs, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        f = frames[idx]
        if f.shape[-1] == 1:
            f = np.repeat(f, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = f
        ax_frames.text(
            i * w + w / 2,
            h + 12,
            f"f{idx}\n{per_frame_rewards[idx]:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample frames", pad=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_topreward_progress(
    dataset_repo_id: str,
    *,
    image_key: str = "observation.images.image",
    output_path: str | None = None,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    num_samples: int = 15,
    fps: float = 2.0,
    reduction: str = "mean",
    episode_indices: list[int] | None = None,
    device: str | None = None,
    torch_dtype: str = "auto",
    attn_implementation: str | None = None,
    push_to_hub: bool = False,
    num_visualizations: int = 0,
    viz_dir: str | None = None,
) -> Path:
    """Apply TOPReward to every episode of ``dataset_repo_id`` and save a parquet."""
    init_logging()

    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    cfg = TOPRewardConfig(
        model_name=model_name,
        num_samples=num_samples,
        fps=fps,
        reduction=reduction,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model = TOPRewardModel(cfg)
    model.eval()

    selected = (
        sorted({int(i) for i in episode_indices})
        if episode_indices is not None
        else list(range(dataset.num_episodes))
    )

    rows: list[dict[str, Any]] = []
    viz_dir_path = Path(viz_dir) if viz_dir else Path(dataset.root) / "topreward_viz"
    viz_remaining = num_visualizations
    for episode_idx in tqdm(selected, desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        num_frames = ep_end - ep_start

        try:
            frames, task = _load_episode_frames(dataset, ep_start, ep_end, image_key)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Episode {episode_idx}: failed to load frames ({e}); skipping")
            continue

        # Mirror what `calculate_rewards(return_all_frames=True)` does internally so we
        # can persist the prefix lengths alongside the rewards.
        n_samples = min(num_samples, num_frames)
        prefix_lengths = sorted({int(length) for length in np.linspace(1, num_frames, n_samples, dtype=int)})

        prefix_rewards: list[float] = []
        for length in prefix_lengths:
            output = model.compute_instruction_reward(frames[:length], task)
            prefix_rewards.append(float(output.reward))

        if cfg.normalize_prefix_rewards:
            prefix_rewards = TOPRewardModel.normalize_rewards(prefix_rewards).tolist()

        per_frame = _interpolate_to_frames(
            np.asarray(prefix_lengths, dtype=np.int64),
            np.asarray(prefix_rewards, dtype=np.float64),
            num_frames,
        )
        for local_idx, value in enumerate(per_frame):
            rows.append(
                {
                    "index": ep_start + local_idx,
                    "episode_index": episode_idx,
                    "frame_index": local_idx,
                    "progress_topreward": float(value),
                }
            )

        if viz_remaining > 0:
            try:
                visualize_topreward_episode(
                    frames=frames,
                    prefix_lengths=np.asarray(prefix_lengths, dtype=np.int64),
                    prefix_rewards=np.asarray(prefix_rewards, dtype=np.float64),
                    per_frame_rewards=per_frame,
                    task=task,
                    output_path=viz_dir_path / f"topreward_ep{episode_idx:04d}.png",
                )
                viz_remaining -= 1
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Episode {episode_idx}: visualization failed ({e})")

    if not rows:
        raise RuntimeError("No episodes were scored — nothing to write.")

    table = pa.Table.from_pylist(rows)
    metadata = {
        b"reward_model": b"topreward",
        b"model_name": cfg.model_name.encode(),
    }
    table = table.replace_schema_metadata(metadata)

    out_path = Path(output_path) if output_path else Path(dataset.root) / "topreward_progress.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    logging.info(f"Saved {len(rows)} rows to {out_path}")

    if push_to_hub:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_path.name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )
        logging.info(f"Uploaded {out_path.name} to {dataset_repo_id}")

    return out_path


def plot_topreward_from_parquet(
    *,
    dataset_repo_id: str,
    parquet_path: str | Path,
    episode_indices: list[int],
    image_key: str = "observation.images.image",
    viz_dir: str | Path = "./topreward_viz",
    num_thumbnails: int = 8,
) -> list[Path]:
    """Re-plot rewards for selected LIBERO episodes from a saved parquet.

    Useful when ``compute_topreward_progress`` was already run and you only want
    new plots without re-invoking the VLM. Prefix samples cannot be reconstructed
    from the per-frame parquet, so they are not drawn here.
    """
    init_logging()
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pq.read_table(parquet_path).to_pandas()
    dataset = LeRobotDataset(dataset_repo_id)
    viz_dir_path = Path(viz_dir)

    written: list[Path] = []
    for episode_idx in episode_indices:
        ep_df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
        if ep_df.empty:
            logging.warning(f"Episode {episode_idx}: no rows in parquet; skipping")
            continue
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        frames, task = _load_episode_frames(dataset, ep_start, ep_end, image_key)

        per_frame = ep_df["progress_topreward"].to_numpy(dtype=np.float32)
        out_path = viz_dir_path / f"topreward_ep{episode_idx:04d}.png"
        visualize_topreward_episode(
            frames=frames,
            prefix_lengths=np.asarray([1, len(frames)], dtype=np.int64),
            prefix_rewards=np.asarray([per_frame[0], per_frame[-1]], dtype=np.float64),
            per_frame_rewards=per_frame,
            task=task,
            output_path=out_path,
            num_thumbnails=num_thumbnails,
        )
        written.append(out_path)
        logging.info(f"Saved plot to {out_path}")
    return written


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply TOPReward zero-shot rewards to a LIBERO LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", required=True, help="HF dataset repo id or local path")
    parser.add_argument(
        "--image-key",
        default="observation.images.image",
        help="Image key for the agentview camera (LIBERO default: observation.images.image)",
    )
    parser.add_argument("--output-path", default=None, help="Where to write the parquet")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--reduction", default="mean", choices=["mean", "sum"])
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated episode indices (default: all)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=0,
        help="Plot the first N processed episodes as PNGs (requires matplotlib).",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Where to write the per-episode plots (default: <dataset_root>/topreward_viz).",
    )
    args = parser.parse_args()

    compute_topreward_progress(
        dataset_repo_id=args.dataset_repo_id,
        image_key=args.image_key,
        output_path=args.output_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        fps=args.fps,
        reduction=args.reduction,
        episode_indices=_parse_indices(args.episode_indices),
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        push_to_hub=args.push_to_hub,
        num_visualizations=args.num_visualizations,
        viz_dir=args.viz_dir,
    )


if __name__ == "__main__":
    main()
