# SO101 Diffusion Policy

A simplified implementation of **Diffusion Policy** for the **SO101 robotic arm**.

This repository is modified from the original Stanford **Diffusion Policy** project and only retains the components required for **SO101 image-based imitation learning**.

---

# Features

Current supported features:

* RGB image policy
* Depth image policy
* RGB-D image policy
* Future action prediction (Future3 / Future6 / Future9)
* SO101 Zarr dataset
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

```
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
├── dp_inference_server.py
├── train.py
├── setup.py
├── README.md
└── conda_environment_nomujoco.yaml
```

---

# Dataset

Datasets are stored on NAS.

```
/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/
```

Available datasets:

```
so101_demo.zarr
so101_rgbd.zarr
so101_rgbd_future3.zarr
so101_rgbd_future6.zarr
so101_rgbd_future9.zarr
```

---

# Checkpoints

Training checkpoints are stored on NAS.

```
/home/itri2026-3090/NAS/115/itri2026/dp_ckpt/
```

Example:

```
dp_ckpt/
├── 2026.07.02
├── 2026.07.08
└── 2026.07.14
```

---

# Project Layout

Repository:

```
~/diffusion_policy
│
├── diffusion_policy/
├── train.py
├── dp_inference_server.py
└── data/
    └── outputs/
```

NAS:

```
~/NAS/115/itri2026/
│
├── dp_dataset
│   ├── so101
│   │   └── zarr
│   └── reactive_diffusion_policy_dataset
│
└── dp_ckpt
    ├── 2026.07.02
    ├── 2026.07.08
    └── 2026.07.14
```

---

# Training

## RGB (Baseline)

```bash
WANDB_MODE=disabled python train.py \
--config-name=train_diffusion_unet_image_workspace \
task=so101_image \
task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/so101_rgbd_future6.zarr \
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

---

## Depth Only

```bash
WANDB_MODE=disabled python train.py \
--config-name=train_diffusion_unet_image_workspace \
task=so101_image \
task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/so101_rgbd_future6.zarr \
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

---

## RGB-D

```bash
WANDB_MODE=disabled python train.py \
--config-name=train_diffusion_unet_image_workspace \
task=so101_image \
task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/so101_rgbd_future6.zarr \
task.obs_mode=rgbd \
task.image_shape=[4,240,320] \
task.in_channels=4 \
training.num_epochs=100 \
training.rollout_every=0 \
training.checkpoint_every=5 \
training.val_every=5
```

---

# Observation Modes

| Mode  | Input       | Channels |
| ----- | ----------- | -------: |
| RGB   | RGB Image   |        3 |
| Depth | Depth Image |        1 |
| RGB-D | RGB + Depth |        4 |

---

# Future Prediction

Switch the dataset path to select different prediction horizons.

| Dataset                 | Description            |
| ----------------------- | ---------------------- |
| so101_rgbd_future3.zarr | Predict 3 future steps |
| so101_rgbd_future6.zarr | Predict 6 future steps |
| so101_rgbd_future9.zarr | Predict 9 future steps |

Example:

```bash
task.dataset_path=/home/itri2026-3090/NAS/115/itri2026/dp_dataset/so101/zarr/so101_rgbd_future6.zarr
```

---

# Inference

Run the inference server:

```bash
python dp_inference_server.py
```

---

# Core Repository

The simplified SO101 version retains only the modules required for image-based imitation learning.

```
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

---

# Reference

Original project:

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**

Modified for the SO101 robotic arm platform.

---

# License

This project is modified from the original Stanford Diffusion Policy repository and follows the original LICENSE.
