import matplotlib.pyplot as plt
import numpy as np
from numpy import float32
import pandas as pd
from datetime import datetime


def result_fig(n_classes, lables, confusion_matrix, history, seed, bs, sigma):
    new_confusion_matrix = np.zeros(shape=(5, 5))
    sum_col_list = np.sum(confusion_matrix, axis=0)
    # sum_row_list = np.sum(confusion_matrix, axis=1)
    # print(sum_row_list, '\n', sum_col_list)
    for row in range(len(confusion_matrix)):
        for col in range(len(confusion_matrix[row])):
            new_confusion_matrix[row][col] = confusion_matrix[row][col] / sum_col_list[row]
                                             # (sum_row_list[row] + sum_col_list[col] - confusion_matrix[row][col])
    print('================== confusion_matrix =====================')
    print(new_confusion_matrix)

    with open('E:\\PythonProject\\CNN-SVM\\denseNet\\Result\\result_seed=' + str(seed) + '_bs=' + str(bs) + '_sigma=' + str(sigma) + '_add.txt', 'w') as f:  # 设置文件对象data.txt
        print('================== confusion_matrix =====================', '\n',
              new_confusion_matrix, file=f)

    normalised_confusion_matrix = np.array(confusion_matrix, dtype=float32) / np.sum(confusion_matrix) * 100
    # print(normalised_confusion_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap='Blues',
        alpha=0.7, vmin=0, vmax=1,
    )
    plt.title("Confusion matrix")
    plt.colorbar(label='colorbar with Normalize')
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, lables, rotation=15, fontsize=10)
    plt.yticks(tick_marks, lables, fontsize=10)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(new_confusion_matrix)):
        for second_index in range(len(new_confusion_matrix[first_index])):
            plt.text(first_index,
                     second_index,
                     "%0.5f" % new_confusion_matrix[first_index][second_index],
                     horizontalalignment='center',
                     verticalalignment='center')
    plt.savefig('E:\\PythonProject\\CNN-SVM\\denseNet\\Pic\\conf_seed=' + str(seed) + '_bs=' + str(bs) + '_sigma=' + str(sigma) + '.png', dpi=200, bbox_inches='tight')
    plt.show()

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training And Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('E:\\PythonProject\\CNN-SVM\\denseNet\\Pic\\acc_seed=' + str(seed) + '_bs=' + str(bs) + '_sigma=' + str(sigma) + '.png', dpi=200, bbox_inches='tight')
    plt.show()  # 以上为画出val_accuracy和accuracy的图

    # 以下为val_loss和loss的图
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training And Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('E:\\PythonProject\\CNN-SVM\\denseNet\\Pic\\loss_seed=' + str(seed) + '_bs=' + str(bs) + '_sigma=' + str(sigma) + '.png', dpi=200, bbox_inches='tight')
    plt.show()

    df = pd.DataFrame(columns=['time', 'epochs', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])  # 列名
    file_path = "E:\\PythonProject\\CNN-SVM\\denseNet\\Result\\result_data_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".csv"
    df.to_csv(file_path, index=False)  # 路径可以根据需要更改
    time = "%s" % datetime.now()  # 获取当前时间
    list = [time, epochs, loss, accuracy, val_loss, val_accuracy]
    data = pd.DataFrame([list])
    data.to_csv(file_path, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

