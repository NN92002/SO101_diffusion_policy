import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = (
    "/home/itri2026-3090/SO101_diffusion_policy/data/outputs/2026.08.04/21.34.04_train_diffusion_unet_image_so101_image/logs.json.txt"
)

OUTPUT_PATH = "loss_curve_0804-4.png"

# -----------------------
# Read log
# -----------------------
df = pd.read_json(LOG_PATH, lines=True)

print("Columns:")
print(df.columns.tolist())

# -----------------------
# Keep validation records
# -----------------------
val_df = df[df["val_loss"].notna()].copy()

print("\nValidation Results:")
print(
    val_df[
        [
            "epoch",
            "train_loss",
            "val_loss",
            "train_action_mse_error",
        ]
    ]
)

# -----------------------
# Best validation model
# -----------------------
best_idx = val_df["val_loss"].idxmin()

best_epoch = int(val_df.loc[best_idx, "epoch"])
best_val = float(val_df.loc[best_idx, "val_loss"])
best_mse = float(val_df.loc[best_idx, "train_action_mse_error"])

print("\n==============================")
print("        Best Model")
print("==============================")
print(f"Epoch              : {best_epoch}")
print(f"Validation Loss    : {best_val:.6f}")
print(f"Train Action MSE   : {best_mse:.8f}")
print("==============================")

# -----------------------
# Plot Loss Curve
# -----------------------
plt.figure(figsize=(10, 6))

plt.plot(
    val_df["epoch"],
    val_df["train_loss"],
    marker="o",
    linewidth=2,
    label="Train Loss",
)

plt.plot(
    val_df["epoch"],
    val_df["val_loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss",
)

# Best point
plt.scatter(
    best_epoch,
    best_val,
    s=180,
    marker="*",
    color="red",
    zorder=5,
    label=f"Best Epoch {best_epoch}",
)

plt.annotate(
    f"Epoch {best_epoch}\nVal={best_val:.4f}",
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 2, best_val + 0.01),
    arrowprops=dict(arrowstyle="->"),
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SO101 Diffusion Policy Loss Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)

print(f"\nFigure saved to: {OUTPUT_PATH}")