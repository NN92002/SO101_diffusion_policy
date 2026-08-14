if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _masked_mean(value, mask):
    """Mean over valid elements only."""
    mask = mask.to(dtype=value.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def compute_task_aware_diffusion_loss(policy, batch, cfg):
    """
    Diffusion epsilon-prediction loss + task-aware x0 reconstruction loss.

    Expected action layout:
        [x, y, z, qx, qy, qz, qw, gripper]

    The auxiliary task losses are computed in normalized action space so the
    different physical units do not dominate one another. They backpropagate
    through the same epsilon prediction used by DDPM.

    To avoid unstable x0 reconstruction at very noisy diffusion steps, the
    auxiliary loss is applied only when timestep <= task_loss_max_timestep.
    """
    # Same normalization path as DiffusionUnetImagePolicy.compute_loss().
    assert 'valid_mask' not in batch
    nobs = policy.normalizer.normalize(batch['obs'])
    nactions = policy.normalizer['action'].normalize(batch['action'])

    batch_size = nactions.shape[0]
    horizon = nactions.shape[1]

    local_cond = None
    global_cond = None
    trajectory = nactions
    cond_data = trajectory

    if policy.obs_as_global_cond:
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = policy.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)
    else:
        this_nobs = dict_apply(
            nobs,
            lambda x: x.reshape(-1, *x.shape[2:])
        )
        nobs_features = policy.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        cond_data = torch.cat([nactions, nobs_features], dim=-1)
        trajectory = cond_data.detach()

    condition_mask = policy.mask_generator(trajectory.shape)

    noise = torch.randn(trajectory.shape, device=trajectory.device)
    bsz = trajectory.shape[0]
    timesteps = torch.randint(
        0,
        policy.noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=trajectory.device
    ).long()

    noisy_trajectory = policy.noise_scheduler.add_noise(
        trajectory, noise, timesteps
    )

    loss_mask = ~condition_mask
    noisy_trajectory[condition_mask] = cond_data[condition_mask]

    pred = policy.model(
        noisy_trajectory,
        timesteps,
        local_cond=local_cond,
        global_cond=global_cond
    )

    pred_type = policy.noise_scheduler.config.prediction_type
    if pred_type != 'epsilon':
        raise ValueError(
            "Task-aware x0 reconstruction currently requires "
            f"prediction_type='epsilon', got {pred_type!r}."
        )

    # ------------------------------------------------------------
    # 1. Original Diffusion Policy epsilon-prediction loss
    # ------------------------------------------------------------
    noise_element_loss = F.mse_loss(pred, noise, reduction='none')
    noise_element_loss = (
        noise_element_loss * loss_mask.to(noise_element_loss.dtype)
    )
    noise_loss = reduce(
        noise_element_loss, 'b ... -> b (...)', 'mean'
    ).mean()

    # ------------------------------------------------------------
    # 2. Reconstruct clean normalized action x0 from epsilon prediction
    #
    # x_t = sqrt(alpha_bar_t) * x0
    #       + sqrt(1-alpha_bar_t) * epsilon
    #
    # x0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon_hat)
    #          / sqrt(alpha_bar_t)
    # ------------------------------------------------------------
    alphas_cumprod = policy.noise_scheduler.alphas_cumprod.to(
        device=trajectory.device,
        dtype=trajectory.dtype
    )
    alpha_bar = alphas_cumprod[timesteps]

    broadcast_shape = [bsz] + [1] * (trajectory.ndim - 1)
    alpha_bar = alpha_bar.reshape(broadcast_shape)

    sqrt_alpha_bar = alpha_bar.sqrt().clamp_min(1e-6)
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).clamp_min(0.0).sqrt()

    pred_x0 = (
        noisy_trajectory - sqrt_one_minus_alpha_bar * pred
    ) / sqrt_alpha_bar

    # Task loss only uses the 8 action dimensions.
    if nactions.shape[-1] != 8:
        raise RuntimeError(
            "Task-aware action loss expects action dimension 8 with layout "
            "[x,y,z,qx,qy,qz,qw,gripper], "
            f"but got {nactions.shape[-1]}."
        )

    pred_action_x0 = pred_x0[..., :8]
    gt_action_x0 = nactions

    action_loss_mask = loss_mask[..., :8]

    # High-noise timesteps can make x0 estimation extremely large.
    max_task_t = int(cfg.training.task_loss_max_timestep)
    timestep_valid = (timesteps <= max_task_t)
    timestep_valid = timestep_valid.reshape(
        [bsz] + [1] * (action_loss_mask.ndim - 1)
    )
    task_mask = action_loss_mask & timestep_valid

    # Position: x, y, z
    position_sq_error = (
        pred_action_x0[..., 0:3] - gt_action_x0[..., 0:3]
    ).square()
    position_loss = _masked_mean(
        position_sq_error,
        task_mask[..., 0:3]
    )

    # Orientation: qx, qy, qz, qw.
    # This is component MSE in normalized action space. Physical geodesic
    # orientation error is still evaluated separately during validation.
    orientation_sq_error = (
        pred_action_x0[..., 3:7] - gt_action_x0[..., 3:7]
    ).square()
    orientation_loss = _masked_mean(
        orientation_sq_error,
        task_mask[..., 3:7]
    )

    # Gripper
    gripper_sq_error = (
        pred_action_x0[..., 7:8] - gt_action_x0[..., 7:8]
    ).square()
    gripper_loss = _masked_mean(
        gripper_sq_error,
        task_mask[..., 7:8]
    )

    weighted_task_loss = (
        cfg.training.task_loss_position_weight * position_loss
        + cfg.training.task_loss_orientation_weight * orientation_loss
        + cfg.training.task_loss_gripper_weight * gripper_loss
    )

    total_loss = (
        cfg.training.task_loss_noise_weight * noise_loss
        + weighted_task_loss
    )

    components = {
        'total_loss': total_loss,
        'noise_loss': noise_loss,
        'position_loss': position_loss,
        'orientation_loss': orientation_loss,
        'gripper_loss': gripper_loss,
        'weighted_task_loss': weighted_task_loss,
        'task_valid_fraction': timestep_valid.float().mean(),
    }
    return total_loss, components


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        use_plateau_scheduler = (
            str(cfg.training.lr_scheduler).lower() == 'reduce_on_plateau'
        )

        if use_plateau_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=cfg.training.lr_reduce_factor,
                patience=cfg.training.lr_reduce_patience,
                threshold=cfg.training.lr_reduce_threshold,
                threshold_mode='abs',
                cooldown=cfg.training.get('lr_reduce_cooldown', 0),
                min_lr=cfg.training.min_lr
            )
        else:
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs) \
                        // cfg.training.gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
            )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = None
        if cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # Metric-based scheduler / early stopping state.
        monitor_key = cfg.training.get('monitor_key', 'val_task_score')
        early_stopping_enabled = cfg.training.get('early_stopping', False)
        best_monitor_value = float('inf')
        early_stop_counter = 0
        stop_training = False

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                train_noise_losses = list()
                train_position_losses = list()
                train_orientation_losses = list()
                train_gripper_losses = list()
                train_task_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute task-aware diffusion loss
                        raw_loss, loss_components = compute_task_aware_diffusion_loss(
                            self.model, batch, cfg
                        )
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            if not use_plateau_scheduler:
                                lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        noise_loss_cpu = loss_components['noise_loss'].item()
                        position_loss_cpu = loss_components['position_loss'].item()
                        orientation_loss_cpu = loss_components['orientation_loss'].item()
                        gripper_loss_cpu = loss_components['gripper_loss'].item()
                        task_loss_cpu = loss_components['weighted_task_loss'].item()

                        tepoch.set_postfix(
                            loss=raw_loss_cpu,
                            noise=noise_loss_cpu,
                            task=task_loss_cpu,
                            refresh=False
                        )
                        train_losses.append(raw_loss_cpu)
                        train_noise_losses.append(noise_loss_cpu)
                        train_position_losses.append(position_loss_cpu)
                        train_orientation_losses.append(orientation_loss_cpu)
                        train_gripper_losses.append(gripper_loss_cpu)
                        train_task_losses.append(task_loss_cpu)

                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'train_noise_loss': noise_loss_cpu,
                            'train_position_loss': position_loss_cpu,
                            'train_orientation_loss': orientation_loss_cpu,
                            'train_gripper_loss': gripper_loss_cpu,
                            'train_task_loss': task_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.optimizer.param_groups[0]['lr']
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                step_log['train_noise_loss'] = np.mean(train_noise_losses)
                step_log['train_position_loss'] = np.mean(train_position_losses)
                step_log['train_orientation_loss'] = np.mean(train_orientation_losses)
                step_log['train_gripper_loss'] = np.mean(train_gripper_losses)
                step_log['train_task_loss'] = np.mean(train_task_losses)

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (
                    env_runner is not None
                    and cfg.training.rollout_every > 0
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                # Run validation and task-space action evaluation.
                # The task-aware training objective is logged as val_loss; the original
                # diffusion-only component is logged as val_noise_loss. Scheduler,
                # early stopping, and best-checkpoint selection still monitor
                # val_task_score by default.
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        # ----- task-aware validation loss -----
                        val_losses = list()
                        val_noise_losses = list()
                        val_position_aux_losses = list()
                        val_orientation_aux_losses = list()
                        val_gripper_aux_losses = list()
                        val_task_aux_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch,
                                    lambda x: x.to(device, non_blocking=True)
                                )
                                loss, val_components = compute_task_aware_diffusion_loss(
                                    self.model, batch, cfg
                                )
                                val_losses.append(loss.detach().cpu())
                                val_noise_losses.append(
                                    val_components['noise_loss'].detach().cpu())
                                val_position_aux_losses.append(
                                    val_components['position_loss'].detach().cpu())
                                val_orientation_aux_losses.append(
                                    val_components['orientation_loss'].detach().cpu())
                                val_gripper_aux_losses.append(
                                    val_components['gripper_loss'].detach().cpu())
                                val_task_aux_losses.append(
                                    val_components['weighted_task_loss'].detach().cpu())
                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break

                        if len(val_losses) > 0:
                            # val_loss is now the TOTAL task-aware training objective.
                            step_log['val_loss'] = torch.stack(val_losses).mean().item()
                            # Keep original diffusion loss separately for diagnosis.
                            step_log['val_noise_loss'] = (
                                torch.stack(val_noise_losses).mean().item())
                            step_log['val_position_loss'] = (
                                torch.stack(val_position_aux_losses).mean().item())
                            step_log['val_orientation_loss'] = (
                                torch.stack(val_orientation_aux_losses).mean().item())
                            step_log['val_gripper_loss'] = (
                                torch.stack(val_gripper_aux_losses).mean().item())
                            step_log['val_task_aux_loss'] = (
                                torch.stack(val_task_aux_losses).mean().item())

                        # ----- denormalized validation action metrics -----
                        # Expected action layout:
                        # [x, y, z, qx, qy, qz, qw, gripper]
                        # predict_action() and dataset action are both in original units.
                        action_metric_sums = {
                            'val_action_mse': 0.0,
                            'val_position_mse': 0.0,
                            'val_position_x_mse': 0.0,
                            'val_position_y_mse': 0.0,
                            'val_position_z_mse': 0.0,
                            'val_position_euclidean_mae_m': 0.0,
                            'val_position_euclidean_rmse_m': 0.0,
                            'val_orientation_quat_mse': 0.0,
                            'val_orientation_angle_mae_deg': 0.0,
                            'val_orientation_angle_rmse_deg': 0.0,
                            'val_gripper_mse': 0.0,
                            'val_gripper_mae_rad': 0.0,
                        }
                        action_metric_batches = 0
                        max_action_eval_steps = cfg.training.get(
                            'max_action_eval_steps', None)

                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Task metrics epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as action_tepoch:
                            for action_eval_idx, action_batch in enumerate(action_tepoch):
                                action_batch = dict_apply(
                                    action_batch,
                                    lambda x: x.to(device, non_blocking=True)
                                )
                                obs_dict = action_batch['obs']
                                gt_action = action_batch['action']

                                result = policy.predict_action(obs_dict)
                                pred_action = result['action_pred']

                                if pred_action.shape != gt_action.shape:
                                    raise RuntimeError(
                                        'Action shape mismatch: '
                                        f'prediction={tuple(pred_action.shape)}, '
                                        f'ground_truth={tuple(gt_action.shape)}'
                                    )
                                if pred_action.shape[-1] != 8:
                                    raise RuntimeError(
                                        'Expected action dimension 8 with layout '
                                        '[x,y,z,qx,qy,qz,qw,gripper], but got '
                                        f'{pred_action.shape[-1]}.'
                                    )

                                action_mse = torch.nn.functional.mse_loss(
                                    pred_action, gt_action)

                                # Position: meters internally, reported in millimeters.
                                pred_position = pred_action[..., 0:3]
                                gt_position = gt_action[..., 0:3]
                                position_error = pred_position - gt_position
                                position_sq_error = position_error.square()
                                position_mse = position_sq_error.mean()
                                position_axis_mse = position_sq_error.mean(dim=(0, 1))
                                position_distance_m = torch.linalg.vector_norm(
                                    position_error, dim=-1)
                                position_euclidean_mae_m = position_distance_m.mean()
                                position_euclidean_rmse_m = torch.sqrt(
                                    position_distance_m.square().mean())

                                # Orientation: quaternion geodesic error in degrees.
                                pred_quat = torch.nn.functional.normalize(
                                    pred_action[..., 3:7], dim=-1, eps=1e-8)
                                gt_quat = torch.nn.functional.normalize(
                                    gt_action[..., 3:7], dim=-1, eps=1e-8)
                                quat_error_positive = (
                                    pred_quat - gt_quat
                                ).square().mean(dim=-1)
                                quat_error_negative = (
                                    pred_quat + gt_quat
                                ).square().mean(dim=-1)
                                quat_mse = torch.minimum(
                                    quat_error_positive,
                                    quat_error_negative
                                ).mean()
                                quat_dot = torch.sum(
                                    pred_quat * gt_quat, dim=-1
                                ).abs().clamp(0.0, 1.0)
                                orientation_angle_deg = torch.rad2deg(
                                    2.0 * torch.acos(quat_dot))
                                orientation_angle_mae_deg = orientation_angle_deg.mean()
                                orientation_angle_rmse_deg = torch.sqrt(
                                    orientation_angle_deg.square().mean())

                                # Gripper index 7 is assumed to be radians.
                                gripper_error = (
                                    pred_action[..., 7] - gt_action[..., 7])
                                gripper_mse = gripper_error.square().mean()
                                gripper_mae_rad = gripper_error.abs().mean()

                                action_metric_sums['val_action_mse'] += action_mse.item()
                                action_metric_sums['val_position_mse'] += position_mse.item()
                                action_metric_sums['val_position_x_mse'] += position_axis_mse[0].item()
                                action_metric_sums['val_position_y_mse'] += position_axis_mse[1].item()
                                action_metric_sums['val_position_z_mse'] += position_axis_mse[2].item()
                                action_metric_sums['val_position_euclidean_mae_m'] += position_euclidean_mae_m.item()
                                action_metric_sums['val_position_euclidean_rmse_m'] += position_euclidean_rmse_m.item()
                                action_metric_sums['val_orientation_quat_mse'] += quat_mse.item()
                                action_metric_sums['val_orientation_angle_mae_deg'] += orientation_angle_mae_deg.item()
                                action_metric_sums['val_orientation_angle_rmse_deg'] += orientation_angle_rmse_deg.item()
                                action_metric_sums['val_gripper_mse'] += gripper_mse.item()
                                action_metric_sums['val_gripper_mae_rad'] += gripper_mae_rad.item()
                                action_metric_batches += 1

                                if max_action_eval_steps is not None and \
                                        action_eval_idx >= (max_action_eval_steps - 1):
                                    break

                        if action_metric_batches == 0:
                            raise RuntimeError(
                                'No validation batches were available for task metrics.')

                        for key, value in action_metric_sums.items():
                            step_log[key] = value / action_metric_batches

                        step_log['val_position_rmse_mm'] = (
                            step_log['val_position_mse'] ** 0.5) * 1000.0
                        step_log['val_position_x_rmse_mm'] = (
                            step_log['val_position_x_mse'] ** 0.5) * 1000.0
                        step_log['val_position_y_rmse_mm'] = (
                            step_log['val_position_y_mse'] ** 0.5) * 1000.0
                        step_log['val_position_z_rmse_mm'] = (
                            step_log['val_position_z_mse'] ** 0.5) * 1000.0
                        step_log['val_position_euclidean_mae_mm'] = (
                            step_log['val_position_euclidean_mae_m'] * 1000.0)
                        step_log['val_position_euclidean_rmse_mm'] = (
                            step_log['val_position_euclidean_rmse_m'] * 1000.0)
                        step_log['val_gripper_rmse_rad'] = (
                            step_log['val_gripper_mse'] ** 0.5)
                        step_log['val_gripper_mae_deg'] = np.degrees(
                            step_log['val_gripper_mae_rad'])
                        step_log['val_gripper_rmse_deg'] = np.degrees(
                            step_log['val_gripper_rmse_rad'])

                        # Dimensionless task score. Lower is better.
                        # Each physical error is divided by a configurable reference
                        # scale before weighted summation, so mm and degrees are not
                        # mixed directly.
                        position_component = (
                            step_log['val_position_euclidean_mae_mm'] /
                            cfg.training.task_score_position_scale_mm
                        )
                        orientation_component = (
                            step_log['val_orientation_angle_mae_deg'] /
                            cfg.training.task_score_orientation_scale_deg
                        )
                        gripper_component = (
                            step_log['val_gripper_mae_deg'] /
                            cfg.training.task_score_gripper_scale_deg
                        )
                        step_log['val_task_position_component'] = position_component
                        step_log['val_task_orientation_component'] = orientation_component
                        step_log['val_task_gripper_component'] = gripper_component
                        step_log['val_task_score'] = (
                            cfg.training.task_score_position_weight * position_component
                            + cfg.training.task_score_orientation_weight * orientation_component
                            + cfg.training.task_score_gripper_weight * gripper_component
                        )

                        if monitor_key not in step_log:
                            raise KeyError(
                                f"Monitor key '{monitor_key}' was not generated. "
                                f"Available validation keys: {sorted(step_log.keys())}"
                            )
                        monitor_value = float(step_log[monitor_key])

                        # Reduce LR according to the task-space metric, not diffusion loss.
                        if use_plateau_scheduler:
                            old_lr = self.optimizer.param_groups[0]['lr']
                            lr_scheduler.step(monitor_value)
                            new_lr = self.optimizer.param_groups[0]['lr']
                            step_log['lr'] = new_lr
                            if new_lr < old_lr:
                                print(
                                    f"[LR Scheduler] {monitor_key} did not improve. "
                                    f"LR reduced: {old_lr:.3e} -> {new_lr:.3e}"
                                )

                        # Early stopping according to the same task-space metric.
                        if early_stopping_enabled:
                            min_delta = cfg.training.early_stopping_min_delta
                            if monitor_value < (best_monitor_value - min_delta):
                                best_monitor_value = monitor_value
                                early_stop_counter = 0
                                print(
                                    f"[Early Stopping] New best {monitor_key}: "
                                    f"{best_monitor_value:.8f}"
                                )
                            else:
                                early_stop_counter += 1
                                print(
                                    f"[Early Stopping] No {monitor_key} improvement: "
                                    f"{early_stop_counter}/"
                                    f"{cfg.training.early_stopping_patience}"
                                )
                                if early_stop_counter >= \
                                        cfg.training.early_stopping_patience:
                                    stop_training = True
                                    print(
                                        f"[Early Stopping] Stop at epoch {self.epoch}. "
                                        f"Best {monitor_key}: "
                                        f"{best_monitor_value:.8f}"
                                    )

                # checkpoint
                # Save last.ckpt at the configured interval.
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                # Evaluate and save top-k checkpoint after every validation.
                # This prevents early stopping before the real best model is saved.
                if monitor_key in step_log:
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                if stop_training:
                    # Save the latest state before leaving the training loop.
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    break

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
