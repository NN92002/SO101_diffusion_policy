import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Task-aware Diffusion Policy losses, task score, and learning rate."
    )
    parser.add_argument("log_path", help="Path to logs.json.txt")
    parser.add_argument(
        "--output",
        default="task_aware_loss_curve.png",
        help="Output loss image path",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_json(log_path, lines=True)

    required = {"epoch", "train_loss"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    epoch_df = (
        df.sort_index()
        .groupby("epoch", as_index=False)
        .last()
        .sort_values("epoch")
    )

    # Validation rows: prefer task score, otherwise fall back to val_loss.
    if "val_task_score" in epoch_df.columns and epoch_df["val_task_score"].notna().any():
        val_df = epoch_df[epoch_df["val_task_score"].notna()].copy()
        monitor_key = "val_task_score"
        monitor_label = "Task Score"
    elif "val_loss" in epoch_df.columns and epoch_df["val_loss"].notna().any():
        val_df = epoch_df[epoch_df["val_loss"].notna()].copy()
        monitor_key = "val_loss"
        monitor_label = "Validation Loss"
    else:
        raise RuntimeError("No validation records were found.")

    best_idx = val_df[monitor_key].idxmin()
    best_row = val_df.loc[best_idx]
    best_epoch = int(best_row["epoch"])
    best_score = float(best_row[monitor_key])

    print("\n==========================================")
    print("      Best Task-aware Validation Model")
    print("==========================================")
    print(f"Epoch                         : {best_epoch}")
    print(f"{monitor_label:<30}: {best_score:.8f}")

    metric_names = [
        ("val_loss", "Total Validation Loss", ".8f"),
        ("val_noise_loss", "Diffusion Noise Loss", ".8f"),
        ("val_task_aux_loss", "Task Auxiliary Loss", ".8f"),
        ("val_position_loss", "Position Aux Loss", ".8f"),
        ("val_orientation_loss", "Orientation Aux Loss", ".8f"),
        ("val_gripper_loss", "Gripper Aux Loss", ".8f"),
        ("val_position_euclidean_mae_mm", "Position MAE (mm)", ".4f"),
        ("val_position_euclidean_rmse_mm", "Position RMSE (mm)", ".4f"),
        ("val_orientation_angle_mae_deg", "Orientation MAE (deg)", ".4f"),
        ("val_orientation_angle_rmse_deg", "Orientation RMSE (deg)", ".4f"),
        ("val_gripper_mae_deg", "Gripper MAE (deg)", ".4f"),
        ("val_gripper_rmse_deg", "Gripper RMSE (deg)", ".4f"),
        ("lr", "Learning Rate", ".3e"),
    ]

    for column, label, fmt in metric_names:
        if column in best_row.index and pd.notna(best_row[column]):
            print(f"{label:<30}: {format(float(best_row[column]), fmt)}")
    print("==========================================")

    # ------------------------------------------------------------
    # Loss curve
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    if "train_loss" in val_df.columns:
        plt.plot(
            val_df["epoch"],
            val_df["train_loss"],
            marker="o",
            linewidth=2,
            label="Train Total Loss",
        )

    if "val_loss" in val_df.columns and val_df["val_loss"].notna().any():
        plt.plot(
            val_df["epoch"],
            val_df["val_loss"],
            marker="o",
            linewidth=2,
            label="Val Total Loss",
        )

    if "val_noise_loss" in val_df.columns and val_df["val_noise_loss"].notna().any():
        plt.plot(
            val_df["epoch"],
            val_df["val_noise_loss"],
            marker="o",
            linewidth=2,
            label="Val Diffusion Noise Loss",
        )

    if "val_task_aux_loss" in val_df.columns and val_df["val_task_aux_loss"].notna().any():
        plt.plot(
            val_df["epoch"],
            val_df["val_task_aux_loss"],
            marker="o",
            linewidth=2,
            label="Val Task Auxiliary Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SO101 Task-aware Diffusion Policy Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Loss figure saved to: {args.output}")

    # ------------------------------------------------------------
    # Task score
    # ------------------------------------------------------------
    if "val_task_score" in val_df.columns and val_df["val_task_score"].notna().any():
        score_output = str(
            Path(args.output).with_name(
                f"{Path(args.output).stem}_task_score{Path(args.output).suffix}"
            )
        )

        plt.figure(figsize=(10, 5))
        plt.plot(
            val_df["epoch"],
            val_df["val_task_score"],
            marker="o",
            linewidth=2,
            label="Validation Task Score",
        )
        plt.scatter(
            best_epoch,
            best_score,
            s=180,
            marker="*",
            zorder=5,
            label=f"Best Epoch {best_epoch}",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Task Score (lower is better)")
        plt.title("Validation Task Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(score_output, dpi=300)
        plt.close()

        print(f"Task-score figure saved to: {score_output}")

    # ------------------------------------------------------------
    # Learning rate
    # ------------------------------------------------------------
    if "lr" in val_df.columns and val_df["lr"].notna().any():
        lr_output = str(
            Path(args.output).with_name(
                f"{Path(args.output).stem}_lr{Path(args.output).suffix}"
            )
        )
        lr_df = val_df[val_df["lr"].notna()]

        plt.figure(figsize=(10, 5))
        plt.plot(lr_df["epoch"], lr_df["lr"], marker="o", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(lr_output, dpi=300)
        plt.close()

        print(f"Learning-rate figure saved to: {lr_output}")


if __name__ == "__main__":
    main()
