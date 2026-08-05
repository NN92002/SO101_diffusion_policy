import json
import matplotlib.pyplot as plt

log_file = "/home/itri2026-3090/SO101_diffusion_policy/data/outputs/2026.08.04/21.34.04_train_diffusion_unet_image_so101_image/logs.json.txt"
epochs = []
mse = []

with open(log_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)

            if "train_action_mse_error" in data:
                epochs.append(data["epoch"])
                mse.append(data["train_action_mse_error"])

        except:
            pass

plt.figure(figsize=(8,5))
plt.plot(epochs, mse, marker='o', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Train Action MSE")
plt.title("Train Action MSE Curve")
plt.grid(True)

# 標出最低點
best_idx = mse.index(min(mse))
plt.scatter(
    epochs[best_idx],
    mse[best_idx],
    s=100,
    label=f"Min MSE = {mse[best_idx]:.6f}"
)

plt.legend()

plt.tight_layout()
plt.savefig("mse_curve_0804-4.png", dpi=300)
plt.show()

print(f"Best Epoch : {epochs[best_idx]}")
print(f"Min MSE    : {mse[best_idx]:.8f}")