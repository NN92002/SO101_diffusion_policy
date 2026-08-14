import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(
    df,
    columns,
    labels,
    ylabel,
    title,
    output_path,
):
    available = [
        (column, label)
        for column, label in zip(columns, labels)
        if column in df.columns and df[column].notna().any()
    ]

    if not available:
        print(f"Skip {title}: no matching metric columns.")
        return

    plt.figure(figsize=(10, 6))

    for column, label in available:
        valid = df[df[column].notna()]
        plt.plot(
            valid["epoch"],
            valid[column],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Task-aware validation action errors against GT."
    )
    parser.add_argument("log_path", help="Path to logs.json.txt")
    parser.add_argument(
        "--output-dir",
        default="task_aware_action_metric_plots",
        help="Output directory",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_json(log_path, lines=True)
    if "epoch" not in raw_df.columns:
        raise KeyError("Column 'epoch' was not found.")

    df = (
        raw_df.sort_index()
        .groupby("epoch", as_index=False)
        .last()
        .sort_values("epoch")
    )

    # Use val_task_score as the primary best-model criterion.
    if "val_task_score" in df.columns and df["val_task_score"].notna().any():
        valid_df = df[df["val_task_score"].notna()]
        best_row = valid_df.loc[valid_df["val_task_score"].idxmin()]
        best_key = "val_task_score"
        best_label = "Validation Task Score"
    elif "val_loss" in df.columns and df["val_loss"].notna().any():
        valid_df = df[df["val_loss"].notna()]
        best_row = valid_df.loc[valid_df["val_loss"].idxmin()]
        best_key = "val_loss"
        best_label = "Validation Loss"
    else:
        best_row = None

    if best_row is not None:
        print("\n==============================================")
        print("   Metrics at Best Task-aware Validation Epoch")
        print("==============================================")
        print(f"Epoch                              : {int(best_row['epoch'])}")
        print(f"{best_label:<35}: {float(best_row[best_key]):.8f}")

        metrics = [
            ("val_loss", "Total Validation Loss"),
            ("val_noise_loss", "Diffusion Noise Loss"),
            ("val_task_aux_loss", "Task Auxiliary Loss"),
            ("val_position_x_rmse_mm", "Position X RMSE (mm)"),
            ("val_position_y_rmse_mm", "Position Y RMSE (mm)"),
            ("val_position_z_rmse_mm", "Position Z RMSE (mm)"),
            ("val_position_euclidean_mae_mm", "Position 3D MAE (mm)"),
            ("val_position_euclidean_rmse_mm", "Position 3D RMSE (mm)"),
            ("val_orientation_angle_mae_deg", "Orientation MAE (deg)"),
            ("val_orientation_angle_rmse_deg", "Orientation RMSE (deg)"),
            ("val_gripper_mae_deg", "Gripper MAE (deg)"),
            ("val_gripper_rmse_deg", "Gripper RMSE (deg)"),
        ]

        for column, label in metrics:
            if column in best_row.index and pd.notna(best_row[column]):
                print(f"{label:<35}: {float(best_row[column]):.8f}")

        print("==============================================")

    # XYZ RMSE
    save_metric_plot(
        df,
        [
            "val_position_x_rmse_mm",
            "val_position_y_rmse_mm",
            "val_position_z_rmse_mm",
        ],
        ["X RMSE", "Y RMSE", "Z RMSE"],
        "RMSE (mm)",
        "Validation End-Effector Position Error by Axis",
        output_dir / "position_axis_rmse_mm.png",
    )

    # 3D position
    save_metric_plot(
        df,
        [
            "val_position_euclidean_mae_mm",
            "val_position_euclidean_rmse_mm",
        ],
        ["3D MAE", "3D RMSE"],
        "Error (mm)",
        "Validation End-Effector 3D Position Error",
        output_dir / "position_3d_error_mm.png",
    )

    # Orientation
    save_metric_plot(
        df,
        [
            "val_orientation_angle_mae_deg",
            "val_orientation_angle_rmse_deg",
        ],
        ["Orientation MAE", "Orientation RMSE"],
        "Error (degrees)",
        "Validation End-Effector Orientation Error",
        output_dir / "orientation_error_deg.png",
    )

    # Gripper
    save_metric_plot(
        df,
        [
            "val_gripper_mae_deg",
            "val_gripper_rmse_deg",
        ],
        ["Gripper MAE", "Gripper RMSE"],
        "Error (degrees)",
        "Validation Gripper Angle Error",
        output_dir / "gripper_error_deg.png",
    )

    # Task score
    save_metric_plot(
        df,
        ["val_task_score"],
        ["Task Score"],
        "Score (lower is better)",
        "Validation Task Score",
        output_dir / "task_score.png",
    )

    # Auxiliary training-objective components
    save_metric_plot(
        df,
        [
            "val_position_loss",
            "val_orientation_loss",
            "val_gripper_loss",
        ],
        [
            "Position Aux Loss",
            "Orientation Aux Loss",
            "Gripper Aux Loss",
        ],
        "Normalized Auxiliary Loss",
        "Task-aware Validation Auxiliary Loss Components",
        output_dir / "task_aux_components.png",
    )


if __name__ == "__main__":
    main()
