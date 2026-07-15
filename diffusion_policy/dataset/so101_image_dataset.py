from typing import Dict
import copy
import torch
import numpy as np

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer
)


class SO101ImageDataset(BaseImageDataset):
    def __init__(
            self,
            zarr_path,
            horizon=16,
            n_obs_steps=2,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            obs_mode="rgb"
        ):

        super().__init__()

        assert obs_mode in ["rgb", "depth", "rgbd"], \
            f"obs_mode must be rgb, depth, or rgbd, got {obs_mode}"

        self.obs_mode = obs_mode

        if obs_mode == "rgb":
            keys = ["rgb", "state", "action"]
        elif obs_mode == "depth":
            keys = ["depth", "state", "action"]
        elif obs_mode == "rgbd":
            keys = ["rgb", "depth", "state", "action"]

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=keys
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )

        train_mask = ~val_mask

        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )

        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):

        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }

        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data,
            last_n_dims=1,
            mode=mode,
            **kwargs
        )

        normalizer['image'] = get_image_range_normalizer()

        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):

        if self.obs_mode == "rgb":
            image = sample["rgb"].astype(np.float32)

        elif self.obs_mode == "depth":
            image = sample["depth"].astype(np.float32)

        elif self.obs_mode == "rgbd":
            rgb = sample["rgb"].astype(np.float32)
            depth = sample["depth"].astype(np.float32)
            image = np.concatenate([rgb, depth], axis=-1)

        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        # [T, H, W, C] -> [T, C, H, W]
        image = np.moveaxis(image, -1, 1) / 255.0

        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        image = image[:self.n_obs_steps]
        agent_pos = agent_pos[:self.n_obs_steps]

        data = {
            'obs': {
                'image': image,
                'agent_pos': agent_pos
            },
            'action': action
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.sampler.sample_sequence(idx)

        data = self._sample_to_data(sample)

        torch_data = dict_apply(
            data,
            torch.from_numpy
        )

        return torch_data


def test():

    zarr_path = (
        '/home/itri2026-3090/'
        'diffusion_policy/data/so101/'
        'so101_rgbd_future6.zarr'
    )

    for obs_mode in ["rgb", "depth", "rgbd"]:

        dataset = SO101ImageDataset(
            zarr_path=zarr_path,
            horizon=16,
            n_obs_steps=2,
            obs_mode=obs_mode
        )

        sample = dataset[0]

        print()
        print("================================")
        print(f"Dataset Test: obs_mode = {obs_mode}")
        print("================================")

        print(
            "image shape:",
            sample['obs']['image'].shape
        )

        print(
            "agent_pos shape:",
            sample['obs']['agent_pos'].shape
        )

        print(
            "action shape:",
            sample['action'].shape
        )

        print(
            "image dtype:",
            sample['obs']['image'].dtype
        )

        print("================================")


if __name__ == "__main__":
    test()