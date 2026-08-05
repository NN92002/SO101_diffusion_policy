
# SO101 Diffusion Policy

> A simplified implementation of **Diffusion Policy** for the **SO101 robotic arm**.
>
> This repository is modified from the original Stanford **Diffusion Policy** and only retains the components required for **image-based imitation learning** on the SO101 platform.

---

# Features

- RGB image policy
- Depth image policy
- RGB-D image policy
- Future action prediction (Future3 / Future6 / Future9)
- SO101 RGB-D to Zarr dataset conversion
- Timestamp synchronization
- Diffusion UNet Image Policy
- SO101 inference server

---

# Version History

| Version | Description |
|---------|-------------|
| v1 | RGB Diffusion Policy |
| v2 | RGB-D Support |
| v3 | Future Action Prediction |

---

# Environment

## Hardware

| Component | Specification |
|----------|---------------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| Driver | 560.35.05 |
| CUDA | 12.6 |

## Software

| Component | Version |
|----------|---------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.9 |
| Framework | PyTorch |
| CUDA | 12.6 |

---

# Installation

## Clone Repository

```bash
git clone <repository_url>
cd SO101_diffusion_policy
```

## Create Conda Environment

```bash
conda env create -f conda_environment_nomujoco.yaml
conda activate diffusion_policy
```

## Install

```bash
pip install -e .
```

Verify installation

```bash
python train.py --help
```

---

# Project Structure

```text
SO101_diffusion_policy/
├── diffusion_policy/
│   ├── codecs/
│   ├── common/
│   ├── config/
│   ├── dataset/
│   ├── env_runner/
│   ├── model/
│   ├── policy/
│   └── workspace/
│
├── convert_so101_rgbd_to_zarr.py
├── dp_inference_server.py
├── train.py
├── setup.py
├── README.md
└── conda_environment_nomujoco.yaml
```

---

# Dataset

## Raw Dataset Structure

Each demonstration episode should contain:

```text
episode_xxxx/
├── image_data.csv
├── robot_state.csv
├── rgb/
│   ├── 000000.png
│   └── ...
└── depth_raw/
    ├── 000000.npy
    └── ...
```

### Required Files

| File | Description |
|------|-------------|
| image_data.csv | Image timestamps |
| robot_state.csv | Robot states and timestamps |
| rgb/ | RGB images |
| depth_raw/ | Raw depth (.npy) |

---

# Dataset Conversion

Convert the raw dataset into Zarr format:

```bash
python convert_so101_rgbd_to_zarr.py     --input /path/to/raw_dataset     --output /path/to/output.zarr     --height 240     --width 320     --min_depth 0.30     --max_depth 0.90     --max_time_diff 0.03     --overwrite
```

## Conversion Arguments

| Argument | Description |
|----------|-------------|
| --input | Raw dataset directory |
| --output | Output Zarr path |
| --height | Output image height |
| --width | Output image width |
| --min_depth | Minimum valid depth (m) |
| --max_depth | Maximum valid depth (m) |
| --max_time_diff | Maximum timestamp difference |
| --overwrite | Overwrite existing dataset |

---

# Future Prediction

Future action labels are generated during dataset conversion.

| Future Step | Time @30 FPS |
|-------------|-------------:|
| Future3 | 0.1 s |
| Future6 | 0.2 s |
| Future9 | 0.3 s |

Example:

```python
future_step = 6
```

The generated dataset may be named:

```text
pick_marker2_rgbd_future6.zarr
```

---

# Converted Dataset Structure

```text
dataset.zarr/
├── data/
│   ├── rgb
│   ├── depth
│   ├── state
│   └── action
├── meta/
│   └── episode_ends
└── conversion_info.json
```

| Array | Shape |
|-------|-------|
| rgb | (N,240,320,3) |
| depth | (N,240,320,1) |
| state | (N,8) |
| action | (N,8) |
| episode_ends | (Episodes,) |

---

# State / Action Format

```text
eef_x
eef_y
eef_z
eef_qx
eef_qy
eef_qz
eef_qw
gripper
```

Both state and action use:

```text
(N,8)
```

---

# Timestamp Synchronization

For every image timestamp, the converter searches for the nearest robot state timestamp.

If the difference exceeds the configured threshold (default 0.03 s), a warning is printed.

---

# Verify Dataset

Check dataset:

```bash
ls dataset.zarr
```

Inspect Zarr arrays:

```bash
python - <<'PY'
import zarr
root=zarr.open("dataset.zarr","r")
print(root["data/rgb"].shape)
print(root["data/depth"].shape)
print(root["data/state"].shape)
print(root["data/action"].shape)
PY
```


# Training

Before training:

```bash
conda activate diffusion_policy
cd ~/SO101_diffusion_policy
```

---

# Training Template

Use this template for all experiments.

```bash
WANDB_MODE=disabled python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  task.dataset_path=/path/to/dataset.zarr \
  task.obs_mode=rgb \
  task.image_shape=[3,240,320] \
  task.in_channels=3 \
  policy.obs_encoder.crop_shape=null \
  policy.obs_encoder.random_crop=False \
  policy.down_dims='[512,1024,2048]' \
  optimizer.lr=1e-4 \
  optimizer.weight_decay=1e-6 \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  training.num_epochs=100 \
  training.rollout_every=0 \
  training.checkpoint_every=5 \
  training.val_every=5
```

---

# Observation Mode

## RGB

```bash
task.obs_mode=rgb
task.image_shape=[3,240,320]
task.in_channels=3
```

## Depth

```bash
task.obs_mode=depth
task.image_shape=[1,240,320]
task.in_channels=1
```

## RGB-D

```bash
task.obs_mode=rgbd
task.image_shape=[4,240,320]
task.in_channels=4
```

---

# Training Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| task.dataset_path | Dataset path | *.zarr |
| task.obs_mode | rgb/depth/rgbd | rgb |
| task.image_shape | Input image shape | [3,240,320] |
| task.in_channels | Input channels | 3 |
| policy.down_dims | Diffusion UNet channels | [512,1024,2048] |
| optimizer.lr | Learning rate | 1e-4 |
| optimizer.weight_decay | Weight decay | 1e-6 |
| dataloader.batch_size | Training batch size | 16 |
| val_dataloader.batch_size | Validation batch size | 16 |
| training.num_epochs | Number of epochs | 100 |
| training.checkpoint_every | Checkpoint interval | 5 |
| training.val_every | Validation interval | 5 |

---

# Common Parameter Tuning

## Change Dataset

```bash
task.dataset_path=/path/to/dataset.zarr
```

---

## Model Size (Diffusion UNet)

Original

```bash
policy.down_dims='[512,1024,2048]'
```

Medium (Recommended)

```bash
policy.down_dims='[256,512,1024]'
```

Small

```bash
policy.down_dims='[128,256,512]'
```

| down_dims | Description |
|-----------|-------------|
| [512,1024,2048] | Original model |
| [256,512,1024] | Reduced model size (recommended) |
| [128,256,512] | Small model |

---

## Learning Rate

```bash
optimizer.lr=1e-4
```

Examples

```text
1e-4
5e-5
1e-5
```

---

## Weight Decay

```bash
optimizer.weight_decay=1e-6
```

Examples

```text
1e-6
1e-4
1e-3
```

---

## Batch Size

```bash
dataloader.batch_size=16
val_dataloader.batch_size=16
```

or

```bash
dataloader.batch_size=8
val_dataloader.batch_size=8
```

---

## Number of Epochs

```bash
training.num_epochs=100
```

Examples

```text
50
100
200
```

---

## Crop

Disable crop

```bash
policy.obs_encoder.crop_shape=null
policy.obs_encoder.random_crop=False
```

Enable random crop

```bash
policy.obs_encoder.crop_shape='[216,288]'
policy.obs_encoder.random_crop=True
```

---

## Checkpoint

```bash
training.checkpoint_every=5
training.val_every=5
```

---

# Example Experiments

## Experiment 1

Reduce model size

```bash
policy.down_dims='[256,512,1024]'
```

## Experiment 2

Increase weight decay

```bash
optimizer.weight_decay=1e-4
```

## Experiment 3

Reduce learning rate

```bash
optimizer.lr=5e-5
```

## Experiment 4

Increase batch size

```bash
dataloader.batch_size=32
val_dataloader.batch_size=32
```

---

# Training Output

```text
data/outputs/
└── YYYY.MM.DD/
    └── HH.MM.SS_train_diffusion_unet_image_so101_image/
        ├── checkpoints/
        │   ├── best.ckpt
        │   └── latest.ckpt
        ├── logs.json.txt
        └── config.yaml
```


# Inference

## Start Inference Server

```bash
conda activate diffusion_policy
cd ~/SO101_diffusion_policy

python dp_inference_server.py \
    --checkpoint /path/to/checkpoints/best.ckpt
```

---

## Typical Checkpoint

```text
data/outputs/
└── YYYY.MM.DD/
    └── HH.MM.SS_train_diffusion_unet_image_so101_image/
        └── checkpoints/
            ├── best.ckpt
            └── latest.ckpt
```

Use:

- **best.ckpt** : Recommended for deployment
- **latest.ckpt** : Latest training checkpoint

---

# Inference Workflow

```text
Camera
   │
   ▼
RGB / Depth Image
   │
   ▼
Image Preprocessing
   │
   ▼
Observation Encoder (ResNet18)
   │
   ▼
Diffusion UNet
   │
   ▼
Future Action Prediction
   │
   ▼
Robot Controller
```

---

# Recommended Training Workflow

1. Collect demonstrations.
2. Convert raw data to Zarr.
3. Train the model.
4. Select the best checkpoint using validation loss.
5. Evaluate on the real robot.
6. Collect failure cases.
7. Retrain with the expanded dataset.

---

# Experiment Suggestions

Recommended order when tuning hyperparameters:

| Step | Parameter |
|------|-----------|
| 1 | `policy.down_dims` |
| 2 | `optimizer.weight_decay` |
| 3 | `optimizer.lr` |
| 4 | `training.num_epochs` |
| 5 | `batch_size` |
| 6 | Early Stopping (future work) |

Only change **one parameter at a time** to make experiments comparable.

---

# Troubleshooting

## Config not found

```text
Cannot find primary config
```

Use:

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace
```

---

## Task not found

```text
Could not find task/lift_image_abs
```

Use:

```bash
task=so101_image
```

---

## Check training configuration

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  --cfg job
```

---

## Verify parameter override

Example:

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=so101_image \
  policy.down_dims='[256,512,1024]' \
  --cfg job | grep -A4 down_dims
```

---

# FAQ

### Which checkpoint should I deploy?

Use **best.ckpt**.

---

### Which model size is recommended?

```text
[256,512,1024]
```

It provides a good balance between model capacity and computational cost.

---

### Which weight decay should I start with?

```text
1e-6 (default)
```

Typical experiments:

```text
1e-4
1e-3
```

---

### How many epochs?

Start with

```text
100
```

and choose the checkpoint with the lowest validation loss.

---

# TODO

- Early Stopping
- EMA tuning
- Multi-camera support
- Reactive Diffusion Policy
- Marker-based observation
- Force / tactile sensing
- Automatic dataset validation
- Configurable future prediction
- ONNX / TensorRT deployment

---

# Citation

If this repository is useful, please cite the original Diffusion Policy paper.

```bibtex
@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and others},
  booktitle={Robotics: Science and Systems},
  year={2023}
}
```

---

# License

This project is built upon the original Stanford Diffusion Policy implementation.

Please follow the corresponding open-source licenses when using this repository.
