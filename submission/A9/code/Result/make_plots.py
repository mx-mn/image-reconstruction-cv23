import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    output_path = Path.cwd() / 'loss_curves'
    metrics_path = Path.cwd() / 'training_log.csv'

    raw_df = pd.read_csv(metrics_path)
    df = raw_df.drop('epoch', axis=1)
    min_val_loss = df[:60]['val_loss'].min()
    ax = df[:60].plot(title=f'Training and Validation Loss \n best val loss:{min_val_loss:.4f}', marker='.', figsize=(16,9), alpha=0.6, xlabel='Epochs')
    plt.tight_layout()
    plt.savefig(output_path / 'MSE_60_epochs.png')

    min_val_loss =  df[60:-1]['val_loss'].min()
    ax = df[60:-1].plot(title=f'Training and Validation Loss \n best val loss:{min_val_loss:.4f}', marker='.', figsize=(16,9), alpha=0.6, xlabel='Epochs')
    plt.tight_layout()
    plt.savefig(output_path / 'extra-training-with-mean-absolute-error.png')

    df = df.drop(range(59,len(df)-1)).reset_index(drop=True)
    df = df.drop(range(40))

    min_val_loss = df['val_loss'].min()
    ax = df.plot(title=f'Last Epochs plus Validation after extra training \n best val loss:{min_val_loss:.4f}', marker='.', figsize=(16,9), alpha=0.6, xlabel='Epochs')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_path / 'Last-epochs-plus-extra-training-with-mean-absolute-error.png')

if __name__ == '__main__':
    main()