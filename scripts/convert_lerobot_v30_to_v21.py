"""Export a local LeRobot v3.0 dataset in the v2.1 layout used by OpenPI.

The conversion is non-destructive: ``--output-root`` must differ from ``--root``.
Video files are split and re-encoded per episode because v3.0 stores multiple
episodes in each video file while v2.1 expects one video per episode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_parquet_tree(root: Path) -> pa.Table:
    paths = sorted(root.glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    tables = [pq.read_table(path) for path in paths]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _write_jsonlines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _episode_stats(row: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _, feature, statistic = key.split("/", maxsplit=2)
        stats.setdefault(feature, {})[statistic] = _jsonable(value)
    return stats


def _replace_feature_type(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {key: _replace_feature_type(item) for key, item in value.items()}
        if converted.get("_type") == "List":
            converted["_type"] = "Sequence"
        return converted
    if isinstance(value, list):
        return [_replace_feature_type(item) for item in value]
    return value


def _use_legacy_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    serialized_features = metadata.get(b"huggingface")
    if serialized_features is None:
        return table
    features = json.loads(serialized_features)
    metadata[b"huggingface"] = json.dumps(_replace_feature_type(features)).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _find_ffmpeg() -> str:
    if executable := shutil.which("ffmpeg"):
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ffmpeg is required to split v3 videos. Install ffmpeg or imageio-ffmpeg."
        ) from exc


def _split_video(
    ffmpeg: str,
    source: Path,
    destination: Path,
    start_s: float,
    end_s: float,
    expected_frames: int,
) -> None:
    if end_s <= start_s:
        raise ValueError(f"Invalid video interval [{start_s}, {end_s}] for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.9f}",
            "-i",
            str(source),
            "-t",
            f"{end_s - start_s:.9f}",
            "-frames:v",
            str(expected_frames),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def convert_dataset(
    root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    skip_videos: bool = False,
) -> None:
    root = root.resolve()
    output_root = output_root.resolve()
    if root == output_root:
        raise ValueError("--output-root must differ from --root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a v3.0 dataset, got {info.get('codebase_version')!r}")

    episodes_table = _read_parquet_tree(root / "meta" / "episodes")
    episode_rows = sorted(episodes_table.to_pylist(), key=lambda item: item["episode_index"])
    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    task_rows = sorted(tasks_table.to_pylist(), key=lambda item: item["task_index"])
    video_keys = sorted(
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    )
    chunk_size = int(info.get("chunks_size", 1000))

    legacy_info = dict(info)
    legacy_info["codebase_version"] = "v2.1"
    legacy_info["data_path"] = V21_DATA_PATH
    legacy_info["video_path"] = V21_VIDEO_PATH if video_keys else None
    legacy_info["total_chunks"] = math.ceil(len(episode_rows) / chunk_size)
    legacy_info["total_videos"] = len(episode_rows) * len(video_keys)
    legacy_info.pop("data_files_size_in_mb", None)
    legacy_info.pop("video_files_size_in_mb", None)

    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "info.json").write_text(
        json.dumps(legacy_info, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(root / "meta" / "stats.json", output_root / "meta" / "stats.json")
    manifest = root / "meta" / "recording_manifest.jsonl"
    if manifest.exists():
        shutil.copy2(manifest, output_root / "meta" / manifest.name)

    _write_jsonlines(output_root / "meta" / "tasks.jsonl", task_rows)
    _write_jsonlines(
        output_root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": row["episode_index"],
                "tasks": row["tasks"],
                "length": row["length"],
            }
            for row in episode_rows
        ],
    )
    _write_jsonlines(
        output_root / "meta" / "episodes_stats.jsonl",
        [
            {"episode_index": row["episode_index"], "stats": _episode_stats(row)}
            for row in episode_rows
        ],
    )

    data_cache: dict[tuple[int, int], pa.Table] = {}
    ffmpeg = None if skip_videos or not video_keys else _find_ffmpeg()
    for position, row in enumerate(episode_rows, start=1):
        episode_index = int(row["episode_index"])
        episode_chunk = episode_index // chunk_size
        data_location = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_location not in data_cache:
            source_data = root / "data" / f"chunk-{data_location[0]:03d}" / f"file-{data_location[1]:03d}.parquet"
            data_cache[data_location] = pq.read_table(source_data)
        table = data_cache[data_location]
        episode_table = table.filter(pc.equal(table["episode_index"], episode_index))
        if episode_table.num_rows != int(row["length"]):
            raise ValueError(
                f"Episode {episode_index} has {episode_table.num_rows} data rows, expected {row['length']}"
            )
        destination_data = output_root / V21_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        # The datasets version pinned by OpenPI recognizes `Sequence`, not the
        # newer Hugging Face `_type: List` metadata stored by LeRobot v3.
        episode_table = _use_legacy_huggingface_metadata(episode_table)
        pq.write_table(episode_table, destination_data, compression="snappy")

        if ffmpeg is not None:
            for video_key in video_keys:
                prefix = f"videos/{video_key}"
                video_chunk = int(row[f"{prefix}/chunk_index"])
                video_file = int(row[f"{prefix}/file_index"])
                source_video = (
                    root
                    / "videos"
                    / video_key
                    / f"chunk-{video_chunk:03d}"
                    / f"file-{video_file:03d}.mp4"
                )
                destination_video = output_root / V21_VIDEO_PATH.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                _split_video(
                    ffmpeg,
                    source_video,
                    destination_video,
                    float(row[f"{prefix}/from_timestamp"]),
                    float(row[f"{prefix}/to_timestamp"]),
                    int(row["length"]),
                )
        print(f"Converted episode {episode_index} ({position}/{len(episode_rows)})", flush=True)

    if skip_videos and video_keys:
        print("Warning: videos were skipped; this output cannot be used for visual policy training.")
    print(f"Wrote LeRobot v2.1 dataset to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Source LeRobot v3.0 dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination LeRobot v2.1 root.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-videos", action="store_true", help="Only for metadata/data conversion tests.")
    args = parser.parse_args()
    convert_dataset(**vars(args))


if __name__ == "__main__":
    main()
