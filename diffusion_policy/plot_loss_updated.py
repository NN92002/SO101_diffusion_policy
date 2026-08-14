import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot train/validation loss and learning rate.')
    parser.add_argument('log_path', help='Path to logs.json.txt')
    parser.add_argument('--output', default='loss_curve.png')
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f'Log file not found: {log_path}')

    df = pd.read_json(log_path, lines=True)
    required = {'epoch', 'train_loss', 'val_loss'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')

    epoch_df = df.sort_index().groupby('epoch', as_index=False).last().sort_values('epoch')
    val_df = epoch_df[epoch_df['val_loss'].notna()].copy()
    if val_df.empty:
        raise RuntimeError('No validation records containing val_loss were found.')

    best_idx = val_df['val_loss'].idxmin()
    best_row = val_df.loc[best_idx]
    best_epoch = int(best_row['epoch'])
    best_val = float(best_row['val_loss'])

    print('\n==============================')
    print('          Best Model')
    print('==============================')
    print(f'Epoch                     : {best_epoch}')
    print(f'Validation Loss           : {best_val:.8f}')
    print(f'Train Loss                : {float(best_row["train_loss"]):.8f}')

    metric_names = [
        ('lr', 'Learning Rate', '.3e'),
        ('val_action_mse', 'Overall Action MSE', '.8f'),
        ('val_position_euclidean_mae_mm', 'Position MAE (mm)', '.4f'),
        ('val_position_euclidean_rmse_mm', 'Position RMSE (mm)', '.4f'),
        ('val_orientation_angle_mae_deg', 'Orientation MAE (deg)', '.4f'),
        ('val_orientation_angle_rmse_deg', 'Orientation RMSE (deg)', '.4f'),
        ('val_gripper_mae_deg', 'Gripper MAE (deg)', '.4f'),
        ('val_gripper_rmse_deg', 'Gripper RMSE (deg)', '.4f'),
    ]
    for column, label, fmt in metric_names:
        if column in best_row.index and pd.notna(best_row[column]):
            print(f'{label:<26}: {format(float(best_row[column]), fmt)}')
    print('==============================')

    plt.figure(figsize=(10, 6))
    plt.plot(val_df['epoch'], val_df['train_loss'], marker='o', linewidth=2, label='Train Loss')
    plt.plot(val_df['epoch'], val_df['val_loss'], marker='o', linewidth=2, label='Validation Loss')
    plt.scatter(best_epoch, best_val, s=180, marker='*', zorder=5, label=f'Best Epoch {best_epoch}')
    plt.annotate(
        f'Epoch {best_epoch}\nVal={best_val:.6f}',
        xy=(best_epoch, best_val),
        xytext=(8, 12),
        textcoords='offset points',
        arrowprops={'arrowstyle': '->'},
    )
    plt.xlabel('Epoch')
    plt.ylabel('Diffusion Noise MSE Loss')
    plt.title('SO101 Diffusion Policy Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    if 'lr' in val_df.columns and val_df['lr'].notna().any():
        lr_output = str(Path(args.output).with_name(f'{Path(args.output).stem}_lr{Path(args.output).suffix}'))
        lr_df = val_df[val_df['lr'].notna()]
        plt.figure(figsize=(10, 5))
        plt.plot(lr_df['epoch'], lr_df['lr'], marker='o', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(lr_output, dpi=300)
        plt.close()
        print(f'Learning-rate figure saved to: {lr_output}')

    print(f'Loss figure saved to: {args.output}')


if __name__ == '__main__':
    main()
