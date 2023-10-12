# Dense Block 简洁版
import os
import random
import time

import tensorflow as tf
from datetime import datetime

import numpy as np
from keras.callbacks import ReduceLROnPlateau
from sklearn import metrics
from keras import layers, losses, optimizers, callbacks, regularizers, initializers
from keras.models import Model
from keras.layers import Input, ZeroPadding1D, LSTM, Multiply
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, BatchNormalization
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.pooling import AveragePooling1D, GlobalAveragePooling1D


from data_process import load_data
from result_plot import result_fig
from add_noise import add_noise, gauss_noise_matrix


def seed_tensorflow(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'WALKING_UPHILL', 'WALKING_DOWNHILL']
labels = np.array(LABELS)


def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    函数从数字索引对输出标签进行编码。
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def conv_block(x, filters, dropout_rate=None):
    # ------------------------------------------
    #  x        #输入张量
    #  filters  #卷积核的个数
    #  dropout  #参数
    # --------------------------------------------
    x = ZeroPadding1D()(x)
    x = Conv1D(filters=filters, kernel_size=3, use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, nb_layers, nb_filters, dropout_rate=None, grop_rate=0):
    # ------------------------------------------
    #   x        #输入张量
    #  nb_filters  #卷积核的个数
    #  nb_layers  # 本dense_block 中 卷积的层数
    #  dropout_rate  #dropout参数
    #  grop_rate # 每次卷积完后，下一个卷积增加的卷积核个数
    # --------------------------------------------
    concat_feat = x
    for _ in range(nb_layers):
        x = conv_block(concat_feat, filters=nb_filters, dropout_rate=dropout_rate)
        concat_feat = layers.Concatenate()([x, concat_feat])

        if grop_rate != 0:
            nb_filters += grop_rate

    return concat_feat


def transition_block(x, out_fliters):
    # ------------------------------------------
    #  transition_block 用于调整通道数和特征图大小，原文中使用压缩比 按比例来调整通道数，如 0.8
    #   改用out_filters 可以自定义调整后通道数，不用一定按照比例来
    #    x  输入张量
    #   out_fliters 调整后的通道数
    # ---------------------------------------------
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(out_fliters, 1, 1, use_bias=False)(x)

    x = AveragePooling1D(2, 2)(x)

    return x


def dense_example(input_shape, classes):
    # ------------------------------------------
    # 搭建 自己 Dense_net 的示例
    # -----------------------------------------
    input = Input(shape=input_shape, name='data')

    # 注意力层
    attention_probs = Dense(8, activation='softmax', name='attention_vec')(input)
    attention_mul = Multiply()([input, attention_probs])
    x = ZeroPadding1D((3, 3))(attention_mul)

    # x = ZeroPadding1D((3, 3))(input)
    x = Conv1D(filters=2, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第一个块
    x = dense_block(x, nb_layers=4, nb_filters=64, dropout_rate=0.2, grop_rate=32)
    x = transition_block(x, 256)

    # 第二个块
    x = dense_block(x, nb_layers=4, nb_filters=128, dropout_rate=0.2, grop_rate=32)
    x = transition_block(x, 512)

    # x = GlobalAveragePooling1D()(x)
    x = LSTM(64)(x)

    x = Dense(classes)(x)
    x = Softmax()(x)
    return Model(input, x)


def train_test():
    train_data, test_data, train_label, test_label = load_data(sigma)

    # 加入噪声后的数据集
    # train_data = gauss_noise_matrix(train_data, sigma)
    # test_data = gauss_noise_matrix(test_data, sigma)

    input_shape = (len(train_data[0]), len(train_data[0][0]))
    classes = 5
    model = dense_example(input_shape, classes)

    loss = losses.SparseCategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=lr,  # 学习率
                                # decay=1e-6  # 衰减率
                                )

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    EarlyStopping1 = callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
    EarlyStopping2 = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False, )
    Reduce = ReduceLROnPlateau(monitor='val_accuracy',
                               factor=1e-6,
                               patience=2,
                               verbose=2,
                               mode='auto')


    CHECK_ROOT = 'EPOCHS-100/checkpoint' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    if not os.path.exists(CHECK_ROOT):
        os.makedirs(CHECK_ROOT)
    filepath = os.path.join(CHECK_ROOT, 'model.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5')
    ModelCheckpoint = callbacks.ModelCheckpoint(filepath,
                                                monitor='val_accuracy',
                                                # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
                                                verbose=2,
                                                # 信息展示模式，0、1或2。为1表示输出epoch模型保存信息，默认为0表示不输出该信息,为2每个epoch输出一行记录
                                                save_best_only=True,  # 当设置为True时，将只保存在验证集上性能最好的模型
                                                mode='max')

    TensorBoard = callbacks.TensorBoard(log_dir='./logs')

    my_callbacks = [
        # EarlyStopping1,
        # EarlyStopping2,
        # Reduce,
        ModelCheckpoint,
        TensorBoard,
    ]

    start_time = datetime.now()

    history = model.fit(train_data,  # 训练集的输入特征
                        train_label,  # 训练集的标签
                        batch_size=bs,  # 每一个batch的大小（批尺寸），即训练一次网络所用的样本数
                        validation_data=(test_data, test_label),  # (测试集的输入特征，测试集的标签）
                        epochs=eps,  # 迭代次数
                        callbacks=my_callbacks  # 回调
                        )

    end_time = datetime.now()

    predict = model.predict(test_data)
    pred_index_total = []
    for pred in predict:
        pred_index = []
        pred_list = pred.tolist()
        index_max = pred_list.index(max(pred_list))
        pred_index.append(index_max)
        pred_index_total.append(np.array(pred_index))

    one_hot_predictions = one_hot(np.array(pred_index_total))
    prediction = one_hot_predictions.argmax(1)
    confusion_matrix = metrics.confusion_matrix(test_label, prediction)
    print(confusion_matrix)

    print('================== 分类报告 =====================')
    print('Classification report for classifier %s:\n%s\n' % (
        classes, metrics.classification_report(test_label, prediction)))

    print('================== 准确率，召回率，F1值，精确率 =====================')
    print('Accuracy score:', accuracy_score(test_label, prediction))
    print('Recall:', recall_score(test_label, prediction, average='macro'))
    print('F1-score:', f1_score(test_label, prediction, average='macro'))
    print('Precision score:', precision_score(test_label, prediction, average='macro'))

    result_fig(classes, labels, confusion_matrix, history, seed, bs, sigma)
    # plot_confuse(model, test_data, test_label, labels)

    print('start_time：', start_time, 'end_time：', end_time, 'all_time：', end_time - start_time)

    with open('E:\\PythonProject\\CNN-SVM\\denseNet\\Result\\result__seed=' + str(seed) + '_bs=' + str(bs) + '_sigma=' + str(sigma) + '.txt', 'w') as f:  # 设置文件对象data.txt
        print(confusion_matrix, '\n',

              '\n', '================== 分类报告 =====================', '\n',
              'Classification report for classifier %s:\n%s\n' % (
               classes, metrics.classification_report(test_label, prediction)), '\n',

              '\n', '================== 准确率，召回率，F1值，精确率 =====================', '\n',
              'Accuracy score:', accuracy_score(test_label, prediction), '\n',
              'Recall:', recall_score(test_label, prediction, average='macro'), '\n',
              'F1-score:', f1_score(test_label, prediction, average='macro'), '\n',
              'Precision score:', precision_score(test_label, prediction, average='macro'), '\n',

              '\n', 'start_time：', start_time, 'end_time：', end_time, 'all_time：', end_time - start_time,
              file=f)


if __name__ == "__main__":
    seed = 100
    lr = 1e-4
    bs = 32
    eps = 100
    sigma = 0

    seed_tensorflow(seed)
    train_test()

    # train_data, test_data, train_label, test_label = load_data(sigma)
    # input_shape = (len(train_data[0]), len(train_data[0][0]))
    # classes = 5
    # model = dense_example(input_shape, classes)
    # model.summary()
    # print('the number of layers in this model:' + str(len(model.layers)))
