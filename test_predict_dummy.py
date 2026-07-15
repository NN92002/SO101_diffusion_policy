import torch
import hydra

ckpt_path = "data/outputs/2026.06.18/15.59.32_train_diffusion_unet_image_so101_image/checkpoints/epoch=0050-val_loss=0.015108.ckpt"

payload = torch.load(ckpt_path, map_location="cpu")
cfg = payload["cfg"]

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace.load_checkpoint(path=ckpt_path)

policy = workspace.ema_model
policy.eval()
policy.cuda()

obs = {
    "image": torch.zeros((1, 2, 3, 240, 320), device="cuda"),
    "agent_pos": torch.zeros((1, 2, 8), device="cuda")
}

with torch.no_grad():
    result = policy.predict_action(obs)

print(result.keys())
for k, v in result.items():
    if torch.is_tensor(v):
        print(k, v.shape)
        print(v.detach().cpu().numpy()[0, :3])
