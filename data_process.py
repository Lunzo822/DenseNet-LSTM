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

    # Sort by sequence depth, by third axis (transposed)
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
    # Numpy will treat the last two dimensions of the matrix as a two-dimensional matrix, where h represents the number of matrices of w * c size. The overall matrix is a stack of w * c matrices on the h dimension

    maximums, minimums = test_data.max(axis=0).max(axis=0), test_data.min(axis=0).min(axis=0)
    test_data = test_data.astype("float32")
    for j in range(test_data.shape[0]):  # Shape calculates the dimensionality of the data
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


# Disrupting data and labels
def shuffle(train_data, train_label):
    index = [i for i in range(len(train_data))]
    np.random.seed(5298)
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]
    return train_data, train_label


def load_data(sigma):
    # import data
    data1, label1 = load_dataset("train/平上下楼")
    data2, label2 = load_dataset("train/上下坡")

    data_test, label_test = load_dataset("test/data_8滑窗")

    # Stack arrays in vertical (row by row) order
    data_train = np.vstack((data1, data2))
    label_train = np.vstack((label1, label2))

    # # Dataset with added noise
    # data_train = gauss_noise_matrix(data_train, sigma)
    # data_test = gauss_noise_matrix(data_test, sigma)

    # data normalization
    data_train = min_max_scale(data_train)
    data_train = data_train.transpose((0, 2, 1))

    data_test = test_min_max_scale(data_test)
    data_test = data_test.transpose((0, 2, 1))      # transposition

    # Disrupting data and labels
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


def load_test_data(sub):
    data_test, label_test = load_dataset(f'运动意图数据集/五名未经训练的受试者测试集数据/new/{sub}')

    # 数据归一化
    data_test = test_min_max_scale(data_test)
    data_test = data_test.transpose((0, 2, 1))      # 转置

    # 打乱数据和标签
    data_shf, label_shf = shuffle(data_test, label_test)

    test_data, test_label = data_shf, label_shf

    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # print(train_data)
    # print(train_label)
    # print(test_data)
    # print(test_label)

    return test_data, test_label

