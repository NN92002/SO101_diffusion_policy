# SO101 Diffusion Policy

A simplified implementation of **Diffusion Policy** for the **SO101 robotic arm**.

This repository is modified from the original Stanford **Diffusion Policy** project and only retains the components required for **SO101 image-based imitation learning**.

---

# Features

Current supported features:

* RGB image policy
* Depth image policy
* RGB-D image policy
* Future action prediction
* SO101 Zarr dataset conversion
* Image and robot-state timestamp synchronization
* Diffusion UNet Image Policy
* SO101 inference server

---

# Version History

| Version | Description                                     |
| ------- | ----------------------------------------------- |
| v1      | RGB Diffusion Policy                            |
| v2      | RGB-D Support                                   |
| v3      | Future Prediction (Future3 / Future6 / Future9) |

---

# Environment

## Hardware

| Component | Specification                   |
| --------- | ------------------------------- |
| GPU       | NVIDIA GeForce RTX 3090 (24 GB) |
| Driver    | 560.35.05                       |
| CUDA      | 12.6                            |

---

## Software

| Component  | Version                                     |
| ---------- | ------------------------------------------- |
| OS         | Ubuntu 22.04 LTS                            |
| Python     | 3.9                                         |
| CUDA       | 12.6                                        |
| Framework  | PyTorch                                     |
| Repository | Customized Diffusion Policy (SO101 Version) |

---

## Conda Environment

Create the environment:

```bash
conda env create -f conda_environment_nomujoco.yaml
```

Activate the environment:

```bash
conda activate diffusion_policy
```

Install the package:

```bash
pip install -e .
```

---

# Project Structure

```text
.
├── diffusion_policy
│   ├── codecs
│   ├── common
│   ├── config
│   ├── dataset
│   ├── env_runner
│   ├── model
│   ├── policy
│   └── workspace
│
├── convert_so101_rgbd_to_zarr.py
├── dp_inference_server.py
├── train.py
├── setup.py
├── README.md
└── conda_environment_nomujoco.yaml
```

---

# Raw Dataset

Raw SO101 demonstration datasets are stored on NAS.

Example:

```text
/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/
```

Each episode must contain the following files and directories:

```text
pick_marker2/
├── episode_0001/
│   ├── image_data.csv
│   ├── robot_state.csv
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── depth_raw/
│       ├── 000000.npy
│       ├── 000001.npy
│       └── ...
│
├── episode_0002/
├── episode_0003/
└── ...
```

Required episode contents:

| File or Directory | Description                          |
| ----------------- | ------------------------------------ |
| `image_data.csv`  | Image timestamps and image paths     |
| `robot_state.csv` | Robot timestamps and robot states    |
| `rgb/`            | RGB images                           |
| `depth_raw/`      | Raw RealSense depth arrays in `.npy` |

The converter synchronizes each RGB-D frame with the nearest robot state using timestamps.

---

# Dataset Conversion

The script `convert_so101_rgbd_to_zarr.py` converts the raw SO101 dataset into the Zarr format required for Diffusion Policy training.

Run the conversion command from the repository root:

```bash
python convert_so101_rgbd_to_zarr.py \
  --input /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/ \
  --output /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr \
  --height 240 \
  --width 320 \
  --min_depth 0.30 \
  --max_depth 0.90 \
  --max_time_diff 0.03 \
  --overwrite
```

If the converter is stored at another location, use its full path:

```bash
python /home/itri2026-3090/SO101_diffusion_policy/convert_so101_rgbd_to_zarr.py \
  --input /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/ \
  --output /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr \
  --height 240 \
  --width 320 \
  --min_depth 0.30 \
  --max_depth 0.90 \
  --max_time_diff 0.03 \
  --overwrite
```

---

## Conversion Arguments

| Argument          | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `--input`         | Root directory containing raw episode folders         |
| `--output`        | Output Zarr dataset path                              |
| `--height`        | Output image height                                   |
| `--width`         | Output image width                                    |
| `--min_depth`     | Minimum depth in meters                               |
| `--max_depth`     | Maximum depth in meters                               |
| `--max_time_diff` | Expected maximum image and robot timestamp difference |
| `--overwrite`     | Delete and replace an existing output dataset         |

Default values:

```text
height        = 240
width         = 320
min_depth     = 0.30 m
max_depth     = 0.90 m
max_time_diff = 0.03 s
```

The `--overwrite` option deletes the existing output Zarr directory before conversion.

Do not use `--overwrite` when an existing dataset must be preserved.

---

## Future Action Setting

The current converter uses:

```python
future_step = 9
```

The action at each frame is generated from the robot state nine image frames into the future:

```python
actions[:-future_step] = states[future_step:]
actions[-future_step:] = states[-1]
```

At 30 FPS:

```text
9 / 30 = 0.3 seconds
```

Therefore, each action represents the robot state approximately `0.3` seconds after the current observation.

Because the current converter uses `future_step = 9`, the recommended output name is:

```text
pick_marker2_rgbd_future9.zarr
```

To generate another future prediction dataset, change the value in the converter:

```python
future_step = 3
```

or:

```python
future_step = 6
```

or:

```python
future_step = 9
```

Approximate future time at 30 FPS:

| Future Step | Approximate Time |
| ----------: | ---------------: |
|           3 |      0.1 seconds |
|           6 |      0.2 seconds |
|           9 |      0.3 seconds |

---

## Converted Zarr Structure

The generated dataset contains:

```text
pick_marker2_rgbd_future9.zarr/
├── data/
│   ├── rgb
│   ├── depth
│   ├── state
│   └── action
│
├── meta/
│   └── episode_ends
│
└── conversion_info.json
```

Dataset arrays:

| Array               | Shape              | Data Type |
| ------------------- | ------------------ | --------- |
| `data/rgb`          | `(N, 240, 320, 3)` | `uint8`   |
| `data/depth`        | `(N, 240, 320, 1)` | `uint8`   |
| `data/state`        | `(N, 8)`           | `float32` |
| `data/action`       | `(N, 8)`           | `float32` |
| `meta/episode_ends` | `(num_episodes,)`  | `int64`   |

---

## State and Action Format

The state and action dimensions are:

```text
eef_x
eef_y
eef_z
eef_qx
eef_qy
eef_qz
eef_qw
joint_6
```

The first seven values describe the end-effector pose:

```text
eef_x
eef_y
eef_z
eef_qx
eef_qy
eef_qz
eef_qw
```

The last value describes the gripper state:

```text
joint_6
```

Both `state` and `action` have eight dimensions:

```text
(N, 8)
```

---

## RGB Processing

RGB images are:

1. Loaded with OpenCV.
2. Converted from BGR to RGB.
3. Resized to `320 × 240`.
4. Stored as `uint8`.

Saved shape:

```text
(N, 240, 320, 3)
```

---

## Depth Processing

Raw RealSense depth arrays are expected to use millimeters.

The converter performs the following operations:

1. Loads the `.npy` depth file.
2. Converts millimeters to meters.
3. Replaces invalid depth values.
4. Clips depth to the configured range.
5. Normalizes depth to `[0, 1]`.
6. Converts normalized depth to `[0, 255]`.
7. Stores the result as `uint8`.

Depth conversion:

```python
depth = depth / 1000.0
```

Depth normalization:

```python
depth = np.clip(depth, min_depth, max_depth)
depth = (depth - min_depth) / (max_depth - min_depth)
depth = depth * 255.0
```

Saved shape:

```text
(N, 240, 320, 1)
```

---

## Timestamp Synchronization

Image timestamps are read from:

```text
image_data.csv
```

Robot timestamps are read from:

```text
robot_state.csv
```

For every image timestamp, the converter finds the nearest robot timestamp.

If the timestamp difference is larger than:

```text
0.03 seconds
```

the converter prints a warning:

```text
[WARN] episode_XXXX: max timestamp diff = X.XXXXs
```

The warning does not stop conversion. It indicates that the image and robot-state streams may not be sufficiently synchronized.

---

## Check Raw Dataset Structure

Before conversion, list all episode CSV files:

```bash
find /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/ \
  -maxdepth 2 \
  -type f \
  \( -name "image_data.csv" -o -name "robot_state.csv" \) \
  | sort
```

Check episode directories:

```bash
find /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/ \
  -maxdepth 1 \
  -type d \
  | sort
```

Check one episode:

```bash
ls -lah \
  /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2/episode_0001
```

Expected result:

```text
image_data.csv
robot_state.csv
rgb
depth_raw
```

---

## Verify Converted Dataset

Check that the Zarr dataset was created:

```bash
ls -lah \
  /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr
```

Check the conversion metadata:

```bash
cat \
  /home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr/conversion_info.json
```

Inspect all array shapes:

```bash
python - <<'PY'
import zarr

dataset_path = (
    "/home/itri2026-3090/NAS/115/itri2026/"
    "dp_dataset/pick_marker2_rgbd_future9.zarr"
)

root = zarr.open(dataset_path, mode="r")

print("rgb:")
print("  shape:", root["data/rgb"].shape)
print("  dtype:", root["data/rgb"].dtype)

print("depth:")
print("  shape:", root["data/depth"].shape)
print("  dtype:", root["data/depth"].dtype)

print("state:")
print("  shape:", root["data/state"].shape)
print("  dtype:", root["data/state"].dtype)

print("action:")
print("  shape:", root["data/action"].shape)
print("  dtype:", root["data/action"].dtype)

print("episode_ends:")
print(root["meta/episode_ends"][:])
PY
```

---

# Dataset

Datasets are stored on NAS.

Original SO101 Zarr directory:

```text
/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/
```

Available example datasets:

```text
so101_demo.zarr
so101_rgbd.zarr
so101_rgbd_future3.zarr
so101_rgbd_future6.zarr
so101_rgbd_future9.zarr
```

New `pick_marker2` dataset:

```text
/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr
```

---

# Checkpoints

Training checkpoints are stored on NAS.

```text
/home/itri2026-3090/NAS/115/itri2026/dp_ckpt/
```

Example:

```text
dp_ckpt/
├── 2026.07.02
├── 2026.07.08
└── 2026.07.14
```

---

# Project Layout

Repository:

```text
~/diffusion_policy
│
├── diffusion_policy/
├── convert_so101_rgbd_to_zarr.py
├── train.py
├── dp_inference_server.py
└── data/
    └── outputs/
```

NAS:

```text
~/NAS/115/itri2026/
│
├── dp_dataset/
│   ├── pick_marker2/
│   ├── pick_marker2_rgbd_future9.zarr
│   ├── so101/
│   │   └── zarr/
│   └── reactive_diffusion_policy_dataset/
│
└── dp_ckpt/
    ├── 2026.07.02/
    ├── 2026.07.08/
    └── 2026.07.14/
```

---

# Training

Before training, activate the environment:

```bash
conda activate diffusion_policy
```

Move to the repository:

```bash
cd /home/itri2026-3090/SO101_diffusion_policy
```

---

## RGB Training

```bash
WANDB_MODE=disabled python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr \
  task.obs_mode=rgb \
  task.image_shape=[3,240,320] \
  task.in_channels=3 \
  policy.obs_encoder.crop_shape=null \
  policy.obs_encoder.random_crop=False \
  training.num_epochs=100 \
  training.rollout_every=0 \
  training.checkpoint_every=5 \
  training.val_every=5
```

RGB input shape:

```text
3 × 240 × 320
```

---

## Depth-Only Training

```bash
WANDB_MODE=disabled python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr \
  task.obs_mode=depth \
  task.image_shape=[1,240,320] \
  task.in_channels=1 \
  policy.obs_encoder.crop_shape=null \
  policy.obs_encoder.random_crop=False \
  training.num_epochs=100 \
  training.rollout_every=0 \
  training.checkpoint_every=5 \
  training.val_every=5
```

Depth input shape:

```text
1 × 240 × 320
```

---

## RGB-D Training

```bash
WANDB_MODE=disabled python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr \
  task.obs_mode=rgbd \
  task.image_shape=[4,240,320] \
  task.in_channels=4 \
  policy.obs_encoder.crop_shape=null \
  policy.obs_encoder.random_crop=False \
  training.num_epochs=100 \
  training.rollout_every=0 \
  training.checkpoint_every=5 \
  training.val_every=5
```

RGB-D input shape:

```text
4 × 240 × 320
```

The four channels consist of:

```text
R
G
B
Depth
```

---

# Observation Modes

| Mode  | Input       | Channels |
| ----- | ----------- | -------: |
| RGB   | RGB image   |        3 |
| Depth | Depth image |        1 |
| RGB-D | RGB + Depth |        4 |

---

# Future Prediction

Future prediction is generated during dataset conversion.

The current `pick_marker2` converter uses:

```python
future_step = 9
```

This produces:

```text
pick_marker2_rgbd_future9.zarr
```

Previous datasets:

| Dataset                   | Description                  |
| ------------------------- | ---------------------------- |
| `so101_rgbd_future3.zarr` | Predict 3 image frames ahead |
| `so101_rgbd_future6.zarr` | Predict 6 image frames ahead |
| `so101_rgbd_future9.zarr` | Predict 9 image frames ahead |

At 30 FPS:

| Dataset | Approximate Prediction Time |
| ------- | --------------------------- |
| Future3 | 0.1 seconds                 |
| Future6 | 0.2 seconds                 |
| Future9 | 0.3 seconds                 |

To switch datasets during training, modify:

```bash
task.dataset_path=/path/to/dataset.zarr
```

Example:

```bash
task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/pick_marker2_rgbd_future9.zarr
```

---

# Training Output

Training results are stored under:

```text
data/outputs/
```

Example:

```text
data/outputs/
└── 2026.07.31/
    └── 11.30.00_train_diffusion_unet_image_so101_image/
        ├── checkpoints/
        │   ├── best.ckpt
        │   └── latest.ckpt
        ├── logs.json.txt
        └── config.yaml
```

Checkpoints are saved according to:

```bash
training.checkpoint_every=5
```

Validation is performed according to:

```bash
training.val_every=5
```

---

# Inference

Run the inference server:

```bash
python dp_inference_server.py
```

Before inference, verify that the checkpoint path in the inference server points to the correct trained model.

Example checkpoint:

```text
/home/itri2026-3090/NAS/115/itri2026/dp_ckpt/2026.07.31/pick_marker2_rgb_future9/checkpoints/best.ckpt
```

---

# Core Repository

The simplified SO101 version retains only the modules required for image-based imitation learning.

```text
diffusion_policy/
├── codecs/
├── common/
├── config/
├── dataset/
├── env_runner/
├── model/
│   ├── common/
│   ├── diffusion/
│   └── vision/
├── policy/
└── workspace/
```

---

# TODO

Future development plans:

* Reactive Diffusion Policy
* Multi-camera observation
* Marker observation
* Force sensing
* Tactile sensing
* Bimanual manipulation
* Contact-rich manipulation
* Configurable future-step conversion argument
* Automatic dataset validation
* Automatic RGB-D visualization
* Dataset timestamp quality report

---

# Reference

Original project:

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**

This repository is modified for the SO101 robotic arm platform.

---

# License

This project is modified from the original Stanford Diffusion Policy repository and follows the original license.
