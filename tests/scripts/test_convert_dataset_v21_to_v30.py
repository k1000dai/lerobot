from pathlib import Path

import pandas as pd
import pytest

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.scripts.convert_dataset_v21_to_v30 import (
    V21,
    convert_data,
    ensure_clean_staging_path,
    legacy_episode_index_from_path,
    replace_root_with_backup,
    validate_contiguous_episode_indices,
)


def _write_minimal_v21_info(root: Path) -> None:
    write_info(
        {
            "codebase_version": V21,
            "fps": 30,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [2]},
                "episode_index": {"dtype": "int64", "shape": [1]},
            },
        },
        root,
    )


def test_legacy_episode_index_from_path() -> None:
    assert legacy_episode_index_from_path(Path("episode_000123.mp4")) == 123

    with pytest.raises(ValueError, match="Expected legacy episode filename"):
        legacy_episode_index_from_path(Path("file-000.mp4"))


def test_validate_contiguous_episode_indices_rejects_gaps() -> None:
    validate_contiguous_episode_indices([0, 1, 2], "Data")

    with pytest.raises(ValueError, match="contiguous and sorted"):
        validate_contiguous_episode_indices([0, 2], "Data")


def test_ensure_clean_staging_path_requires_force(tmp_path: Path) -> None:
    staging = tmp_path / "dataset_v30"
    staging.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_clean_staging_path(staging, force=False)

    ensure_clean_staging_path(staging, force=True)
    assert not staging.exists()


def test_replace_root_with_backup_refuses_existing_backup(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    new_root = tmp_path / "dataset_v30"
    old_root = tmp_path / "dataset_old"
    root.mkdir()
    new_root.mkdir()
    old_root.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing backup"):
        replace_root_with_backup(root, new_root, old_root)

    assert root.exists()
    assert new_root.exists()
    assert old_root.exists()


def test_convert_data_orders_by_episode_index_column(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    new_root = tmp_path / "converted"
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    _write_minimal_v21_info(root)

    pd.DataFrame({"episode_index": [1, 1], "observation.state": [[1.0, 1.0], [1.1, 1.1]]}).to_parquet(
        data_dir / "episode_000000.parquet"
    )
    pd.DataFrame({"episode_index": [0], "observation.state": [[0.0, 0.0]]}).to_parquet(
        data_dir / "episode_000999.parquet"
    )

    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb=100)

    assert [episode["episode_index"] for episode in episodes_metadata] == [0, 1]
    assert episodes_metadata[0]["dataset_from_index"] == 0
    assert episodes_metadata[0]["dataset_to_index"] == 1
    assert episodes_metadata[1]["dataset_from_index"] == 1
    assert episodes_metadata[1]["dataset_to_index"] == 3

    converted = pd.read_parquet(new_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0))
    assert converted["episode_index"].tolist() == [0, 1, 1]


def test_convert_data_rejects_non_contiguous_episode_indices(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    new_root = tmp_path / "converted"
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    _write_minimal_v21_info(root)

    pd.DataFrame({"episode_index": [0], "observation.state": [[0.0, 0.0]]}).to_parquet(
        data_dir / "episode_000000.parquet"
    )
    pd.DataFrame({"episode_index": [2], "observation.state": [[2.0, 2.0]]}).to_parquet(
        data_dir / "episode_000002.parquet"
    )

    with pytest.raises(ValueError, match="contiguous and sorted"):
        convert_data(root, new_root, data_file_size_in_mb=100)
