#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert SO101 raw RGB-D episodes to a Future6 Zarr dataset.

Alignment modes:
- timestamp: match every image to the nearest robot timestamp (original behavior).
- id: match image rows and robot rows by a shared frame/sample ID.

Main features:
- Future offset is configurable and defaults to 6 frames (0.20 s at 30 Hz).
- Long idle tails at the end of each episode are automatically trimmed.
- The final future_step observations are dropped by default instead of padded
  with the final pose, reducing artificial no-op labels.
- Image/robot timestamps are cleaned and validated.
- Misaligned frames can be dropped, rejected, or kept with a warning.
- Quaternion sequences are normalized and q/-q sign flips are removed.
- Episodes are written to Zarr one at a time to avoid high RAM usage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


STATE_COLS = [
    "eef_x", "eef_y", "eef_z",
    "eef_qx", "eef_qy", "eef_qz", "eef_qw",
    "joint_6",
]


@dataclass
class PreparedEpisode:
    name: str
    directory: Path
    image_df: pd.DataFrame
    states: np.ndarray
    timestamps: np.ndarray
    time_diff: np.ndarray
    original_frames: int
    aligned_frames: int
    idle_frames_removed: int
    trailing_idle_seconds: float
    alignment_method: str
    alignment_details: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--future_step", type=int, default=6)
    parser.add_argument(
        "--tail_mode", choices=["drop", "pad"], default="drop",
        help="drop removes the final future_step observations; pad repeats the last state.",
    )

    parser.add_argument(
        "--align_by",
        choices=["timestamp", "id"],
        default="timestamp",
        help=(
            "timestamp: nearest image/robot timestamp matching; "
            "id: exact matching by frame/sample ID."
        ),
    )
    parser.add_argument(
        "--image_id_col",
        default=None,
        help=(
            "Image CSV ID column used when --align_by id. "
            "If omitted, a common ID column is auto-detected."
        ),
    )
    parser.add_argument(
        "--robot_id_col",
        default=None,
        help=(
            "Robot CSV ID column used when --align_by id. "
            "If omitted, a common ID column is auto-detected."
        ),
    )
    parser.add_argument(
        "--id_duplicate_mode",
        choices=["error", "first"],
        default="error",
        help="How duplicated IDs are handled in ID alignment.",
    )

    parser.add_argument("--max_time_diff", type=float, default=0.03)
    parser.add_argument(
        "--alignment_mode", choices=["drop", "error", "warn"], default="drop"
    )

    parser.add_argument("--min_depth", type=float, default=0.30)
    parser.add_argument("--max_depth", type=float, default=0.90)

    parser.add_argument(
        "--trim_idle", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--idle_min_seconds", type=float, default=1.0)
    parser.add_argument("--idle_keep_seconds", type=float, default=0.30)
    parser.add_argument(
        "--idle_translation_threshold", type=float, default=0.0008,
        help="Per-frame XYZ movement threshold in meters.",
    )
    parser.add_argument(
        "--idle_rotation_threshold", type=float, default=0.008,
        help="Per-frame orientation change threshold in radians.",
    )
    parser.add_argument(
        "--idle_gripper_threshold", type=float, default=0.002,
        help="Per-frame gripper change threshold.",
    )
    parser.add_argument(
        "--idle_confirm_frames", type=int, default=5,
        help="Neighborhood used to suppress one-frame noise in idle detection.",
    )
    parser.add_argument("--min_episode_frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.future_step <= 0:
        raise ValueError("future_step must be positive")
    if args.max_time_diff <= 0:
        raise ValueError("max_time_diff must be positive")
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("Require 0 < min_depth < max_depth")
    if args.idle_min_seconds < 0 or args.idle_keep_seconds < 0:
        raise ValueError("Idle times cannot be negative")
    if args.idle_confirm_frames <= 0:
        raise ValueError("idle_confirm_frames must be positive")


def read_rgb(path: Path, resize_hw: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = resize_hw
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8, copy=False)


def read_depth(
    path: Path,
    resize_hw: Tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Cannot read depth file: {path}")

    depth = np.load(str(path)).astype(np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError(f"No valid depth values: {path}")

    median_raw = float(np.median(valid))
    if median_raw < 10.0:
        raise ValueError(
            f"Depth appears to already be in meters: {path}, median={median_raw:.4f}"
        )

    depth /= 1000.0
    depth[~np.isfinite(depth)] = max_depth
    depth[depth <= 0] = max_depth
    depth[depth > max_depth] = max_depth

    h, w = resize_hw
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.rint(np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    return depth[..., None]


def clean_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: List[str],
    name: str,
) -> pd.DataFrame:
    """Validate and clean a CSV dataframe.

    Only columns listed in numeric_columns are converted with pd.to_numeric.
    Path columns such as rgb_path and depth_path must remain strings.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    invalid_numeric = [
        column for column in numeric_columns if column not in required_columns
    ]
    if invalid_numeric:
        raise ValueError(
            f"{name}: numeric columns are not included in required_columns: "
            f"{invalid_numeric}"
        )

    cleaned = df.copy()

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Keep rgb_path/depth_path as strings. Empty strings are treated as missing.
    for column in required_columns:
        if column not in numeric_columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()
            cleaned.loc[cleaned[column] == "", column] = pd.NA

    cleaned = cleaned.dropna(subset=required_columns)
    cleaned = cleaned.sort_values("timestamp")
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="first")
    return cleaned.reset_index(drop=True)



ID_COLUMN_CANDIDATES = [
    "frame_id",
    "frame",
    "id",
    "image_id",
    "sample_id",
    "sample",
    "seq",
    "sequence",
    "index",
]


def resolve_id_column(
    df: pd.DataFrame,
    requested: str | None,
    csv_name: str,
) -> str:
    """Resolve an ID column for ID-based alignment."""
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"{csv_name}: requested ID column '{requested}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    lower_to_original = {str(column).lower(): column for column in df.columns}
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        f"{csv_name}: cannot auto-detect an ID column. "
        f"Tried {ID_COLUMN_CANDIDATES}. "
        f"Available columns: {list(df.columns)}. "
        "Use --image_id_col / --robot_id_col explicitly."
    )


def normalize_id_series(series: pd.Series, name: str) -> pd.Series:
    """Normalize IDs so numeric-looking values compare consistently."""
    result = series.astype("string").str.strip()
    result = result.replace("", pd.NA)

    if result.isna().any():
        raise ValueError(f"{name}: missing/empty IDs found")

    numeric = pd.to_numeric(result, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=np.float64)
        rounded = np.rint(values)
        if np.all(np.isclose(values, rounded, rtol=0.0, atol=1e-9)):
            return pd.Series(
                [str(int(value)) for value in rounded],
                index=series.index,
                dtype="string",
            )

    return result


def align_indices_by_id(
    image_df: pd.DataFrame,
    robot_df: pd.DataFrame,
    image_id_col: str,
    robot_id_col: str,
    duplicate_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return matched image/robot row indices using exact normalized IDs."""
    image_ids = normalize_id_series(
        image_df[image_id_col], f"image_data.csv:{image_id_col}"
    )
    robot_ids = normalize_id_series(
        robot_df[robot_id_col], f"robot_state.csv:{robot_id_col}"
    )

    image_dup = image_ids.duplicated(keep=False)
    robot_dup = robot_ids.duplicated(keep=False)

    if duplicate_mode == "error":
        if image_dup.any():
            examples = image_ids[image_dup].head(10).tolist()
            raise ValueError(
                f"image_data.csv duplicated IDs in '{image_id_col}': {examples}"
            )
        if robot_dup.any():
            examples = robot_ids[robot_dup].head(10).tolist()
            raise ValueError(
                f"robot_state.csv duplicated IDs in '{robot_id_col}': {examples}"
            )

    image_work = pd.DataFrame(
        {"_row": np.arange(len(image_df), dtype=np.int64), "_id": image_ids}
    )
    robot_work = pd.DataFrame(
        {"_row": np.arange(len(robot_df), dtype=np.int64), "_id": robot_ids}
    )

    if duplicate_mode == "first":
        image_work = image_work.drop_duplicates(subset=["_id"], keep="first")
        robot_work = robot_work.drop_duplicates(subset=["_id"], keep="first")

    robot_lookup = dict(zip(robot_work["_id"], robot_work["_row"]))

    matched_image_rows = []
    matched_robot_rows = []
    missing_robot_ids = []

    for image_row, image_id in zip(image_work["_row"], image_work["_id"]):
        robot_row = robot_lookup.get(image_id)
        if robot_row is None:
            missing_robot_ids.append(str(image_id))
            continue
        matched_image_rows.append(int(image_row))
        matched_robot_rows.append(int(robot_row))

    if not matched_image_rows:
        raise ValueError(
            "ID alignment found zero matching IDs between image_data.csv "
            "and robot_state.csv"
        )

    image_id_set = set(image_work["_id"].tolist())
    extra_robot_ids = [
        str(value)
        for value in robot_work.loc[
            ~robot_work["_id"].isin(image_id_set), "_id"
        ].head(20).tolist()
    ]

    info = {
        "image_id_col": str(image_id_col),
        "robot_id_col": str(robot_id_col),
        "matched_ids": int(len(matched_image_rows)),
        "unmatched_image_ids": int(len(missing_robot_ids)),
        "unmatched_robot_ids": int(
            (~robot_work["_id"].isin(image_id_set)).sum()
        ),
        "unmatched_image_id_examples": missing_robot_ids[:20],
        "unmatched_robot_id_examples": extra_robot_ids,
    }

    return (
        np.asarray(matched_image_rows, dtype=np.int64),
        np.asarray(matched_robot_rows, dtype=np.int64),
        info,
    )



def nearest_indices(image_ts: np.ndarray, robot_ts: np.ndarray) -> np.ndarray:
    if len(robot_ts) < 2:
        raise ValueError("robot_state.csv must contain at least two valid rows")

    indices = np.searchsorted(robot_ts, image_ts)
    indices = np.clip(indices, 1, len(robot_ts) - 1)
    left = indices - 1
    right = indices
    left_diff = np.abs(image_ts - robot_ts[left])
    right_diff = np.abs(image_ts - robot_ts[right])
    return np.where(left_diff <= right_diff, left, right)


def rgb_path(ep_dir: Path, row: pd.Series) -> Path:
    value = row.get("rgb_path")
    if pd.isna(value):
        raise ValueError(f"{ep_dir.name}: empty rgb_path")
    path = Path(str(value))
    if not path.exists():
        path = ep_dir / "rgb" / path.name
    return path


def depth_path(ep_dir: Path, row: pd.Series) -> Path:
    if "depth_path" in row.index and not pd.isna(row["depth_path"]):
        path = Path(str(row["depth_path"]))
        if not path.exists():
            path = ep_dir / "depth_raw" / path.name
        return path
    stem = Path(str(row["rgb_path"])).stem
    return ep_dir / "depth_raw" / f"{stem}.npy"


def normalize_quaternions(states: np.ndarray) -> np.ndarray:
    result = states.astype(np.float32, copy=True)
    quaternions = result[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("Invalid zero/NaN quaternion found")

    quaternions /= norms[:, None]
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0

    result[:, 3:7] = quaternions.astype(np.float32)
    return result


def quaternion_step_angles(quaternions: np.ndarray) -> np.ndarray:
    if len(quaternions) < 2:
        return np.empty((0,), dtype=np.float64)
    q1 = quaternions[:-1].astype(np.float64)
    q2 = quaternions[1:].astype(np.float64)
    dots = np.abs(np.sum(q1 * q2, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def expand_motion(motion: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or motion.size == 0:
        return motion.astype(bool)
    kernel = np.ones(window, dtype=np.int32)
    return np.convolve(motion.astype(np.int32), kernel, mode="same") > 0


def detect_idle_tail(
    states: np.ndarray,
    timestamps: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[int, int, float]:
    """Return retained end index (exclusive), removed frames, idle seconds."""
    count = len(states)
    if count < 2:
        return count, 0, 0.0

    translation = np.linalg.norm(np.diff(states[:, 0:3], axis=0), axis=1)
    rotation = quaternion_step_angles(states[:, 3:7])
    gripper = np.abs(np.diff(states[:, 7]))

    moving_transition = (
        (translation > args.idle_translation_threshold)
        | (rotation > args.idle_rotation_threshold)
        | (gripper > args.idle_gripper_threshold)
    )
    moving_transition = expand_motion(
        moving_transition,
        args.idle_confirm_frames,
    )

    moving_frame = np.zeros(count, dtype=bool)
    moving_frame[1:] = moving_transition
    moving_frame[:-1] |= moving_transition
    moving_indices = np.flatnonzero(moving_frame)

    if moving_indices.size == 0:
        # Do not delete a whole episode automatically.
        return count, 0, 0.0

    last_motion = int(moving_indices[-1])
    idle_seconds = float(timestamps[-1] - timestamps[last_motion])
    if idle_seconds < args.idle_min_seconds:
        return count, 0, idle_seconds

    keep_until_time = float(timestamps[last_motion]) + args.idle_keep_seconds
    end = int(np.searchsorted(timestamps, keep_until_time, side="right"))
    end = max(end, min(count, last_motion + 2))
    end = min(end, count)
    return end, count - end, idle_seconds


def prepare_episode(ep_dir: Path, args: argparse.Namespace) -> PreparedEpisode:
    image_csv = ep_dir / "image_data.csv"
    robot_csv = ep_dir / "robot_state.csv"
    image_raw = pd.read_csv(image_csv)
    robot_raw = pd.read_csv(robot_csv)
    original_frames = len(image_raw)

    # timestamp remains required in both modes:
    # - timestamp mode uses it for matching.
    # - id mode uses it only for timing diagnostics and idle-tail duration.
    image_required = ["timestamp", "rgb_path"]
    robot_required = ["timestamp"] + STATE_COLS

    # Preserve potential ID columns while cleaning the required fields.
    image_df = clean_dataframe(
        image_raw,
        required_columns=image_required,
        numeric_columns=["timestamp"],
        name=image_csv.name,
    )
    robot_df = clean_dataframe(
        robot_raw,
        required_columns=robot_required,
        numeric_columns=["timestamp"] + STATE_COLS,
        name=robot_csv.name,
    )

    if len(image_df) < 2 or len(robot_df) < 2:
        raise ValueError("Too few valid CSV rows")

    alignment_details: Dict[str, object] = {}

    if args.align_by == "timestamp":
        image_ts = image_df["timestamp"].to_numpy(dtype=np.float64)
        robot_ts = robot_df["timestamp"].to_numpy(dtype=np.float64)

        indices = nearest_indices(image_ts, robot_ts)
        time_diff = np.abs(image_ts - robot_ts[indices])

        bad = time_diff > args.max_time_diff
        bad_count = int(np.count_nonzero(bad))
        if bad_count:
            message = (
                f"{ep_dir.name}: {bad_count}/{len(time_diff)} frames exceed "
                f"max_time_diff={args.max_time_diff:.4f}s; "
                f"max={float(np.max(time_diff)):.4f}s"
            )
            if args.alignment_mode == "error":
                raise RuntimeError(message)
            if args.alignment_mode == "drop":
                keep = ~bad
                image_df = image_df.loc[keep].reset_index(drop=True)
                image_ts = image_ts[keep]
                indices = indices[keep]
                time_diff = time_diff[keep]
                print(f"[ALIGN-DROP] {message}")
            else:
                print(f"[ALIGN-WARN] {message}")

        aligned_frames = len(image_df)
        if aligned_frames < 2:
            raise ValueError("Too few aligned frames remain")

        states = robot_df.iloc[indices][STATE_COLS].to_numpy(dtype=np.float32)

        alignment_details = {
            "method": "timestamp",
            "max_time_diff_threshold_s": float(args.max_time_diff),
            "time_diff_policy": str(args.alignment_mode),
            "matched_frames": int(aligned_frames),
        }

    elif args.align_by == "id":
        image_id_col = resolve_id_column(
            image_df, args.image_id_col, image_csv.name
        )
        robot_id_col = resolve_id_column(
            robot_df, args.robot_id_col, robot_csv.name
        )

        image_rows, robot_rows, id_info = align_indices_by_id(
            image_df=image_df,
            robot_df=robot_df,
            image_id_col=image_id_col,
            robot_id_col=robot_id_col,
            duplicate_mode=args.id_duplicate_mode,
        )

        # Keep image order. This is important because FutureN is defined along
        # the temporally ordered aligned sequence.
        order = np.argsort(image_rows, kind="stable")
        image_rows = image_rows[order]
        robot_rows = robot_rows[order]

        image_df = image_df.iloc[image_rows].reset_index(drop=True)
        matched_robot_df = robot_df.iloc[robot_rows].reset_index(drop=True)

        image_ts = image_df["timestamp"].to_numpy(dtype=np.float64)
        matched_robot_ts = matched_robot_df["timestamp"].to_numpy(dtype=np.float64)
        time_diff = np.abs(image_ts - matched_robot_ts)

        aligned_frames = len(image_df)
        if aligned_frames < 2:
            raise ValueError("Too few ID-aligned frames remain")

        states = matched_robot_df[STATE_COLS].to_numpy(dtype=np.float32)

        # In ID mode time_diff is diagnostic only. It never changes matching.
        over_dt = int(np.count_nonzero(time_diff > args.max_time_diff))
        if over_dt:
            print(
                f"[ID-DT-WARN] {ep_dir.name}: "
                f"{over_dt}/{aligned_frames} ID-matched frames differ by more than "
                f"{args.max_time_diff:.4f}s; max={float(np.max(time_diff)):.4f}s. "
                "Frames are kept because --align_by id is active."
            )

        alignment_details = {
            "method": "id",
            **id_info,
            "duplicate_mode": str(args.id_duplicate_mode),
            "time_diff_is_diagnostic_only": True,
            "frames_over_max_time_diff": over_dt,
        }

        print(
            f"[ALIGN-ID] {ep_dir.name}: "
            f"{image_id_col} <-> {robot_id_col}, "
            f"matched={aligned_frames}, "
            f"missing_image_ids={id_info['unmatched_image_ids']}, "
            f"extra_robot_ids={id_info['unmatched_robot_ids']}"
        )

    else:
        raise ValueError(f"Unsupported align_by: {args.align_by}")

    states = normalize_quaternions(states)

    removed = 0
    idle_seconds = 0.0
    if args.trim_idle:
        end, removed, idle_seconds = detect_idle_tail(states, image_ts, args)
        if removed > 0:
            image_df = image_df.iloc[:end].reset_index(drop=True)
            states = states[:end]
            image_ts = image_ts[:end]
            time_diff = time_diff[:end]

    required = args.min_episode_frames + (
        args.future_step if args.tail_mode == "drop" else 0
    )
    if len(image_df) < required:
        raise ValueError(
            f"Too few frames after trimming: {len(image_df)}, required={required}"
        )

    return PreparedEpisode(
        name=ep_dir.name,
        directory=ep_dir,
        image_df=image_df,
        states=states,
        timestamps=image_ts,
        time_diff=time_diff,
        original_frames=original_frames,
        aligned_frames=aligned_frames,
        idle_frames_removed=removed,
        trailing_idle_seconds=idle_seconds,
        alignment_method=args.align_by,
        alignment_details=alignment_details,
    )


def build_labels(
    prepared: PreparedEpisode,
    future_step: int,
    tail_mode: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    states = prepared.states
    if tail_mode == "drop":
        count = len(states) - future_step
        images = prepared.image_df.iloc[:count].reset_index(drop=True)
        observations = states[:count].copy()
        actions = states[future_step:].copy()
    else:
        images = prepared.image_df.copy()
        observations = states.copy()
        actions = np.empty_like(states)
        actions[:-future_step] = states[future_step:]
        actions[-future_step:] = states[-1]

    if len(images) != len(observations) or len(observations) != len(actions):
        raise RuntimeError("Internal observation/action length mismatch")
    return images, observations, actions


def create_store(output: Path, height: int, width: int):
    root = zarr.open(str(output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    rgb = data.create_dataset(
        "rgb", shape=(0, height, width, 3),
        chunks=(64, height, width, 3), dtype="uint8", compressor=compressor,
    )
    depth = data.create_dataset(
        "depth", shape=(0, height, width, 1),
        chunks=(64, height, width, 1), dtype="uint8", compressor=compressor,
    )
    state = data.create_dataset(
        "state", shape=(0, len(STATE_COLS)),
        chunks=(1024, len(STATE_COLS)), dtype="float32", compressor=compressor,
    )
    action = data.create_dataset(
        "action", shape=(0, len(STATE_COLS)),
        chunks=(1024, len(STATE_COLS)), dtype="float32", compressor=compressor,
    )
    return meta, rgb, depth, state, action


def append(zarray, values: np.ndarray) -> Tuple[int, int]:
    start = int(zarray.shape[0])
    end = start + int(values.shape[0])
    zarray.resize((end,) + tuple(zarray.shape[1:]))
    zarray[start:end] = values
    return start, end


def convert_episode(
    prepared: PreparedEpisode,
    args: argparse.Namespace,
    rgb_array,
    depth_array,
    state_array,
    action_array,
) -> Dict[str, object]:
    image_df, states, actions = build_labels(
        prepared, args.future_step, args.tail_mode
    )
    count = len(image_df)

    rgbs = np.empty((count, args.height, args.width, 3), dtype=np.uint8)
    depths = np.empty((count, args.height, args.width, 1), dtype=np.uint8)

    for out_index, (_, row) in enumerate(
        tqdm(image_df.iterrows(), total=count, desc=prepared.name)
    ):
        rgbs[out_index] = read_rgb(
            rgb_path(prepared.directory, row), (args.height, args.width)
        )
        depths[out_index] = read_depth(
            depth_path(prepared.directory, row),
            (args.height, args.width),
            args.min_depth,
            args.max_depth,
        )

    starts_ends = [
        append(rgb_array, rgbs),
        append(depth_array, depths),
        append(state_array, states.astype(np.float32, copy=False)),
        append(action_array, actions.astype(np.float32, copy=False)),
    ]
    starts = {item[0] for item in starts_ends}
    ends = {item[1] for item in starts_ends}
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError("Inconsistent Zarr append offsets")

    start = starts_ends[0][0]
    end = starts_ends[0][1]
    return {
        "episode": prepared.name,
        "original_image_frames": int(prepared.original_frames),
        "aligned_frames_before_idle_trim": int(prepared.aligned_frames),
        "idle_frames_removed": int(prepared.idle_frames_removed),
        "trailing_idle_seconds_detected": float(prepared.trailing_idle_seconds),
        "frames_after_idle_trim_before_future": int(len(prepared.image_df)),
        "future_tail_frames_removed": (
            int(args.future_step) if args.tail_mode == "drop" else 0
        ),
        "saved_frames": int(count),
        "alignment_method": prepared.alignment_method,
        "alignment_details": prepared.alignment_details,
        "mean_time_diff_s": float(np.mean(prepared.time_diff)),
        "max_time_diff_s": float(np.max(prepared.time_diff)),
        "zarr_start": int(start),
        "zarr_end": int(end),
    }


def find_episodes(root: Path) -> List[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("episode_")
        and (path / "image_data.csv").exists()
        and (path / "robot_state.csv").exists()
        and (path / "rgb").exists()
        and (path / "depth_raw").exists()
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite")
        shutil.rmtree(output)

    episodes = find_episodes(input_root)
    if not episodes:
        raise RuntimeError(f"No valid episode_* directories in {input_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    meta, rgb, depth, state, action = create_store(
        output, args.height, args.width
    )

    episode_ends: List[int] = []
    infos: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for ep_dir in episodes:
        try:
            prepared = prepare_episode(ep_dir, args)
            if prepared.idle_frames_removed:
                print(
                    f"[IDLE-TRIM] {ep_dir.name}: "
                    f"idle={prepared.trailing_idle_seconds:.2f}s, "
                    f"removed={prepared.idle_frames_removed}, "
                    f"remaining={len(prepared.image_df)} before Future{args.future_step}"
                )
            else:
                print(
                    f"[IDLE-KEEP] {ep_dir.name}: "
                    f"trailing_idle={prepared.trailing_idle_seconds:.2f}s"
                )

            info = convert_episode(prepared, args, rgb, depth, state, action)
            infos.append(info)
            episode_ends.append(int(info["zarr_end"]))
            print(
                f"[OK] {ep_dir.name}: saved={info['saved_frames']}, "
                f"max_dt={info['max_time_diff_s']:.4f}s"
            )
        except Exception as exc:
            failures.append({
                "episode": ep_dir.name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[FAIL] {ep_dir.name}: {type(exc).__name__}: {exc}")

    if not infos:
        shutil.rmtree(output, ignore_errors=True)
        raise RuntimeError("All episodes failed; output removed")

    episode_ends_array = np.asarray(episode_ends, dtype=np.int64)
    meta.create_dataset("episode_ends", data=episode_ends_array, dtype="int64")

    conversion_info = {
        "input_root": str(input_root),
        "output_path": str(output),
        "rgb_shape": [args.height, args.width, 3],
        "depth_shape": [args.height, args.width, 1],
        "depth_unit_raw": "uint16 millimeters",
        "depth_unit_saved": "uint8 normalized",
        "min_depth_m": float(args.min_depth),
        "max_depth_m": float(args.max_depth),
        "state_cols": STATE_COLS,
        "action_definition": f"action[t] = state[t + {args.future_step}]",
        "future_step": int(args.future_step),
        "future_time_at_30hz_s": float(args.future_step / 30.0),
        "tail_mode": args.tail_mode,
        "align_by": args.align_by,
        "alignment_mode": args.alignment_mode,
        "image_id_col_requested": args.image_id_col,
        "robot_id_col_requested": args.robot_id_col,
        "id_duplicate_mode": args.id_duplicate_mode,
        "max_time_diff_s": float(args.max_time_diff),
        "idle_trimming": {
            "enabled": bool(args.trim_idle),
            "idle_min_seconds": float(args.idle_min_seconds),
            "idle_keep_seconds": float(args.idle_keep_seconds),
            "translation_threshold_m_per_frame": float(
                args.idle_translation_threshold
            ),
            "rotation_threshold_rad_per_frame": float(
                args.idle_rotation_threshold
            ),
            "gripper_threshold_per_frame": float(
                args.idle_gripper_threshold
            ),
            "confirm_frames": int(args.idle_confirm_frames),
        },
        "num_episode_directories_found": len(episodes),
        "num_episodes_saved": len(infos),
        "num_episodes_failed": len(failures),
        "total_saved_frames": int(rgb.shape[0]),
        "total_idle_frames_removed": int(
            sum(item["idle_frames_removed"] for item in infos)
        ),
        "total_future_tail_frames_removed": int(
            sum(item["future_tail_frames_removed"] for item in infos)
        ),
        "episode_ends": episode_ends_array.tolist(),
        "episodes": infos,
        "failed_episodes": failures,
    }

    with open(output / "conversion_info.json", "w", encoding="utf-8") as file:
        json.dump(conversion_info, file, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Output:", output)
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    print("State shape:", state.shape)
    print("Action shape:", action.shape)
    print("Episode ends:", episode_ends_array)
    print("Episodes saved:", len(infos))
    print("Episodes failed:", len(failures))
    print("Alignment:", args.align_by)
    print("Idle frames removed:", conversion_info["total_idle_frames_removed"])
    print(
        "Future tail frames removed:",
        conversion_info["total_future_tail_frames_removed"],
    )
    print(
        f"Action label: Future{args.future_step} "
        f"(~{args.future_step / 30.0:.3f}s at 30 Hz)"
    )

    if failures:
        print("\nFailed episodes:")
        for item in failures:
            print(f"  {item['episode']}: {item['error']}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)
