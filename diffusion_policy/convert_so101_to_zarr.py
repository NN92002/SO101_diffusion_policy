import os
import shutil
import argparse
import json
from pathlib import Path

import cv2
import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm


STATE_COLS = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "joint_6",
]


def read_image(path, resize_hw=(240, 320)):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # cv2: BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = resize_hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)


def find_nearest_indices(image_ts, robot_ts):
    """
    image_ts: (N,)
    robot_ts: (M,)
    return nearest robot index for each image timestamp
    """
    indices = np.searchsorted(robot_ts, image_ts)

    indices = np.clip(indices, 1, len(robot_ts) - 1)

    left = indices - 1
    right = indices

    left_diff = np.abs(image_ts - robot_ts[left])
    right_diff = np.abs(image_ts - robot_ts[right])

    nearest = np.where(left_diff <= right_diff, left, right)

    return nearest


def convert_episode(ep_dir, resize_hw=(240, 320), max_time_diff=0.03):
    ep_dir = Path(ep_dir)

    image_csv = ep_dir / "image_data.csv"
    robot_csv = ep_dir / "robot_state.csv"

    if not image_csv.exists():
        raise FileNotFoundError(f"Missing {image_csv}")
    if not robot_csv.exists():
        raise FileNotFoundError(f"Missing {robot_csv}")

    image_df = pd.read_csv(image_csv)
    robot_df = pd.read_csv(robot_csv)

    image_ts = image_df["timestamp"].to_numpy(dtype=np.float64)
    robot_ts = robot_df["timestamp"].to_numpy(dtype=np.float64)

    nearest_idx = find_nearest_indices(image_ts, robot_ts)
    time_diff = np.abs(image_ts - robot_ts[nearest_idx])

    if np.max(time_diff) > max_time_diff:
        print(f"[WARN] {ep_dir.name}: max timestamp diff = {np.max(time_diff):.4f}s")

    imgs = []
    states = []

    for i, row in tqdm(image_df.iterrows(), total=len(image_df), desc=ep_dir.name):
        rgb_path = Path(row["rgb_path"])

        # 如果 csv 裡是舊機器的絕對路徑，改用目前 episode 底下的 rgb/檔名
        if not rgb_path.exists():
            rgb_path = ep_dir / "rgb" / Path(row["rgb_path"]).name

        img = read_image(rgb_path, resize_hw=resize_hw)
        imgs.append(img)

        r = robot_df.iloc[nearest_idx[i]]
        state = r[STATE_COLS].to_numpy(dtype=np.float32)
        states.append(state)

    imgs = np.stack(imgs, axis=0).astype(np.uint8)
    states = np.stack(states, axis=0).astype(np.float32)

    # 第一版 action 直接用對齊後的 EEF pose + gripper
    actions = states.copy()

    info = {
        "episode": ep_dir.name,
        "num_frames": len(imgs),
        "max_time_diff": float(np.max(time_diff)),
        "mean_time_diff": float(np.mean(time_diff)),
    }

    return imgs, states, actions, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="raw dataset root, e.g. /home/itri2026-3090/pick_marker1")
    parser.add_argument("--output", type=str, required=True,
                        help="output zarr path, e.g. /home/itri2026-3090/diffusion_policy/data/so101/so101_demo.zarr")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max_time_diff", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists():
        if args.overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    episode_dirs = sorted([
        p for p in input_root.iterdir()
        if p.is_dir() and (p / "image_data.csv").exists() and (p / "robot_state.csv").exists()
    ])

    if len(episode_dirs) == 0:
        raise RuntimeError(f"No valid episodes found in {input_root}")

    all_imgs = []
    all_states = []
    all_actions = []
    episode_ends = []
    infos = []

    total = 0

    for ep_dir in episode_dirs:
        imgs, states, actions, info = convert_episode(
            ep_dir,
            resize_hw=(args.height, args.width),
            max_time_diff=args.max_time_diff
        )

        all_imgs.append(imgs)
        all_states.append(states)
        all_actions.append(actions)

        total += len(imgs)
        episode_ends.append(total)
        infos.append(info)

    all_imgs = np.concatenate(all_imgs, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(output_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    data_group.create_dataset(
        "img",
        data=all_imgs,
        chunks=(64, args.height, args.width, 3),
        dtype="uint8",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
    )

    data_group.create_dataset(
        "state",
        data=all_states,
        chunks=(1024, all_states.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
    )

    data_group.create_dataset(
        "action",
        data=all_actions,
        chunks=(1024, all_actions.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
    )

    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
    )

    with open(output_path / "conversion_info.json", "w") as f:
        json.dump({
            "input_root": str(input_root),
            "output_path": str(output_path),
            "image_shape": [args.height, args.width, 3],
            "state_cols": STATE_COLS,
            "num_episodes": len(episode_dirs),
            "total_frames": int(total),
            "episode_ends": episode_ends.tolist(),
            "episodes": infos,
        }, f, indent=2)

    print("\nDone.")
    print("Output:", output_path)
    print("img shape:", all_imgs.shape)
    print("state shape:", all_states.shape)
    print("action shape:", all_actions.shape)
    print("episode_ends:", episode_ends)


if __name__ == "__main__":
    main()