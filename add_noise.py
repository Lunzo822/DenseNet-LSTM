import numpy as np
import math


def wgn(x, snr):  # 输出为高斯白噪声
    '''
    程序中用hist()检查噪声是否是高斯分布，psd()检查功率谱密度是否为常数。
    '''
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def add_noise(train_data, test_data):
    snr = 6
    n1 = wgn(train_data, snr)
    n2 = wgn(test_data, snr)
    train_data_noise = train_data + n1  # 增加了6dBz信噪比噪声的信号
    test_data_noise = test_data + n2  # 增加了6dBz信噪比噪声的信号
    print('train_data信噪比：', 10 * math.log10(sum(train_data_noise ** 2) / sum(n1 ** 2)))  # 验算信噪比
    print('test_data信噪比：', 10 * math.log10(sum(test_data_noise ** 2) / sum(n2 ** 2)))  # 验算信噪比

    return train_data_noise, test_data_noise


def gauss_noise_matrix(matrix, sigma):
    # 1. 定义一个与多维矩阵等大的高斯噪声矩阵
    mu = 0
    channel_size = len(matrix)
    height = len(matrix[0])
    width = len(matrix[0][0])
    noise_matrix = np.random.normal(mu, sigma, size=[channel_size, height, width]).astype(
        np.float32)  # 这里在生成噪声矩阵的同时将其元素数据类型转换为float32
    # print("noise_matrix_element_type: {}".format(type(noise_matrix[0][0][0]))) # numpy.float32
    # print(noise_matrix[0][0])  # 这里为了方便观察，只输出了第一个channel的第一行元素

    # 2. 与原来的多维矩阵相加，即可达到添加高斯噪声的效果
    matrix = matrix + noise_matrix

    # 3. 输出添加噪声后的矩阵
    # print(">>>>>>>>>added gaussain noise with method 2")
    # print(matrix[0][0])  # 这里为了方便观察，只输出了第一个channel的第一行元素

    return matrix

