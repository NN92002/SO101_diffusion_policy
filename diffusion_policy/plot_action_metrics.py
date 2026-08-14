import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(df, columns, labels, ylabel, title, output_path):
    available = [(c, l) for c, l in zip(columns, labels) if c in df.columns and df[c].notna().any()]
    if not available:
        print(f'Skip {title}: no matching metric columns.')
        return

    plt.figure(figsize=(10, 6))
    for column, label in available:
        valid = df[df[column].notna()]
        plt.plot(valid['epoch'], valid[column], marker='o', linewidth=2, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Figure saved to: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot denormalized action errors against validation GT.')
    parser.add_argument('log_path', help='Path to logs.json.txt')
    parser.add_argument('--output-dir', default='action_metric_plots')
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f'Log file not found: {log_path}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_json(log_path, lines=True)
    if 'epoch' not in raw_df.columns:
        raise KeyError("Column 'epoch' was not found.")
    df = raw_df.sort_index().groupby('epoch', as_index=False).last().sort_values('epoch')

    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        best_row = df[df['val_loss'].notna()].loc[df[df['val_loss'].notna()]['val_loss'].idxmin()]
        print('\n====================================')
        print(' Metrics at Best Validation Epoch')
        print('====================================')
        print(f"Epoch                         : {int(best_row['epoch'])}")
        print(f"Validation Loss               : {float(best_row['val_loss']):.8f}")
        metrics = [
            ('val_action_mse', 'Overall Action MSE'),
            ('val_position_x_rmse_mm', 'Position X RMSE (mm)'),
            ('val_position_y_rmse_mm', 'Position Y RMSE (mm)'),
            ('val_position_z_rmse_mm', 'Position Z RMSE (mm)'),
            ('val_position_euclidean_mae_mm', 'Position 3D MAE (mm)'),
            ('val_position_euclidean_rmse_mm', 'Position 3D RMSE (mm)'),
            ('val_orientation_quat_mse', 'Quaternion MSE'),
            ('val_orientation_angle_mae_deg', 'Orientation MAE (deg)'),
            ('val_orientation_angle_rmse_deg', 'Orientation RMSE (deg)'),
            ('val_gripper_mse', 'Gripper MSE'),
            ('val_gripper_mae_deg', 'Gripper MAE (deg)'),
            ('val_gripper_rmse_deg', 'Gripper RMSE (deg)'),
        ]
        for column, label in metrics:
            if column in best_row.index and pd.notna(best_row[column]):
                print(f'{label:<30}: {float(best_row[column]):.8f}')
        print('====================================')

    save_metric_plot(df,
        ['val_position_x_rmse_mm','val_position_y_rmse_mm','val_position_z_rmse_mm'],
        ['X RMSE','Y RMSE','Z RMSE'], 'RMSE (mm)',
        'End-Effector Position Error by Axis', output_dir / 'position_axis_rmse_mm.png')

    save_metric_plot(df,
        ['val_position_euclidean_mae_mm','val_position_euclidean_rmse_mm'],
        ['3D MAE','3D RMSE'], 'Error (mm)',
        'End-Effector 3D Position Error', output_dir / 'position_3d_error_mm.png')

    save_metric_plot(df,
        ['val_orientation_angle_mae_deg','val_orientation_angle_rmse_deg'],
        ['Orientation MAE','Orientation RMSE'], 'Error (degrees)',
        'End-Effector Orientation Error', output_dir / 'orientation_error_deg.png')

    save_metric_plot(df,
        ['val_gripper_mae_deg','val_gripper_rmse_deg'],
        ['Gripper MAE','Gripper RMSE'], 'Error (degrees)',
        'Gripper Angle Error', output_dir / 'gripper_error_deg.png')

    save_metric_plot(df,
        ['val_action_mse','val_orientation_quat_mse','val_gripper_mse'],
        ['Overall Action MSE','Quaternion MSE','Gripper MSE'], 'MSE',
        'Model Output vs Ground-Truth MSE', output_dir / 'action_mse.png')


if __name__ == '__main__':
    main()
