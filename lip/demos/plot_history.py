import matplotlib.pyplot as plt
import pandas as pd

from lip.utils.paths import RESULTS_DIR

if __name__ == '__main__':

    model: str = 'inception_pre_trained'

    # OPEN FILES
    accuracy_history: pd.DataFrame = pd.read_csv(RESULTS_DIR / model / 'accuracy_history.csv')
    loss_history: pd.DataFrame = pd.read_csv(RESULTS_DIR / model / 'loss_history.csv')

    best_accuracy_row = accuracy_history[accuracy_history['val'] == accuracy_history['val'].max()]
    best_accuracy_epoch = best_accuracy_row.index.item()
    best_accuracy_value = best_accuracy_row['val'].item()
    print(f'Best accuracy achieved at epoch {best_accuracy_epoch}: {best_accuracy_value:.3f}')

    fig, (ax_loss, ax_acc) = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f'Model: Inception Pre-Trained')

    ax_loss.plot(loss_history['train'], label='training', color='orange')
    ax_loss.plot(loss_history['val'], label='validation', color='blue')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epochs')

    ax_acc.plot(accuracy_history['train'], label='training', color='orange')
    ax_acc.plot(accuracy_history['val'], label='validation', color='blue')
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epochs')

    plt.legend()
    plt.show()
