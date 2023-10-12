from pandas import read_csv
import numpy as np
from numpy import dstack
import os

from add_noise import gauss_noise_matrix


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


def load_group(filepath, filenames):
    loaded = list()
    for name in filenames:
        data = load_file(filepath + name)
        loaded.append(data)

    # 按序列深度排序，按第三轴(转置)
    loaded = dstack(loaded)
    return loaded


def load_dataset(suffix=""):
    filepath = '../运动意图数据集/加入上下坡/' + suffix + '/'

    filenames = list()

    filenames += ['STHipAngle' + '.txt', 'STKneeAngle' + '.txt', 'STShankAngle' + '.txt']
    filenames += ['SWHipAngle' + '.txt', 'SWKneeAngle' + '.txt', 'SWShankAngle' + '.txt']

    X = load_group(filepath, filenames)
    y = load_file(filepath + 'label' + '.txt')
    return X, y


def min_max_scale(test_data):
    # Numpy会将该矩阵的后两维看成一个二维矩阵，h则代表w*c大小的矩阵的个数，整体矩阵是w*c矩阵在h维度上的堆叠

    maximums, minimums = test_data.max(axis=0).max(axis=0), test_data.min(axis=0).min(axis=0)
    test_data = test_data.astype("float32")
    for j in range(test_data.shape[0]):  # shape求的是数据维数
        for i in range(test_data.shape[2]):
            test_data[j, :, i] = (test_data[j, :, i] - minimums[i]) / (maximums[i] - minimums[i])
    return test_data


def test_min_max_scale(data):
    maximums = [59, 9, 21, 74, 18, 36]
    minimums = [-35, -85, -64, -29, -218, -176]
    data = data.astype("float32")
    for j in range(data.shape[0]):
        for i in range(data.shape[2]):
            data[j, :, i] = (data[j, :, i] - minimums[i]) / (maximums[i] - minimums[i])
    return data


# 打乱数据和标签
def shuffle(train_data, train_label):
    index = [i for i in range(len(train_data))]
    np.random.seed(5298)
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]
    return train_data, train_label


def load_data(sigma):
    # 导入数据
    data1, label1 = load_dataset("train/平上下楼")
    data2, label2 = load_dataset("train/上下坡")

    data_test, label_test = load_dataset("test/data_8滑窗")

    # 垂直（按行）顺序堆叠数组
    data_train = np.vstack((data1, data2))
    label_train = np.vstack((label1, label2))

    # # 加入噪声后的数据集
    # data_train = gauss_noise_matrix(data_train, sigma)
    # data_test = gauss_noise_matrix(data_test, sigma)

    # 数据归一化
    data_train = min_max_scale(data_train)
    data_train = data_train.transpose((0, 2, 1))

    data_test = test_min_max_scale(data_test)
    data_test = data_test.transpose((0, 2, 1))      # 转置

    # 打乱数据和标签
    data_sh, label_sh = shuffle(data_train, label_train)
    data_shf, label_shf = shuffle(data_test, label_test)

    train_data, train_label = data_sh, label_sh
    test_data, test_label = data_shf, label_shf

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # print(train_data)
    # print(train_label)
    # print(test_data)
    # print(test_label)

    return train_data, test_data, train_label, test_label

