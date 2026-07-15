import torch
import hydra
from omegaconf import OmegaConf

ckpt_path = "data/outputs/2026.06.18/15.59.32_train_diffusion_unet_image_so101_image/checkpoints/epoch=0050-val_loss=0.015108.ckpt"

payload = torch.load(ckpt_path, map_location="cpu")
cfg = payload["cfg"]

print("workspace target:", cfg._target_)
print("policy target:", cfg.policy._target_)

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)

workspace.load_checkpoint(path=ckpt_path)

policy = workspace.ema_model
policy.eval()
policy.cuda()

print("Loaded policy OK")
print(type(policy))
