import numpy as np
import math


def wgn(x, snr):  # Output is Gaussian white noise
    '''
    In the program, use hist() to check whether the noise is Gaussian distribution, and psd() to check whether the power spectral density is constant.
    '''
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def add_noise(train_data, test_data):
    snr = 6
    n1 = wgn(train_data, snr)
    n2 = wgn(test_data, snr)
    train_data_noise = train_data + n1  # Signal with added 6dBz signal-to-noise ratio noise
    test_data_noise = test_data + n2  # Signal with added 6dBz signal-to-noise ratio noise
    print('train_data SNR：', 10 * math.log10(sum(train_data_noise ** 2) / sum(n1 ** 2)))  # Checking the signal-to-noise ratio
    print('test_data SNR：', 10 * math.log10(sum(test_data_noise ** 2) / sum(n2 ** 2)))  # Checking the signal-to-noise ratio

    return train_data_noise, test_data_noise


def gauss_noise_matrix(matrix, sigma):
    # 1. Define a Gaussian noise matrix as large as a multidimensional matrix
    mu = 0
    channel_size = len(matrix)
    height = len(matrix[0])
    width = len(matrix[0][0])
    noise_matrix = np.random.normal(mu, sigma, size=[channel_size, height, width]).astype(
        np.float32)  # Here, while generating the noise matrix, the element data type is converted to float32
    # print("noise_matrix_element_type: {}".format(type(noise_matrix[0][0][0]))) # numpy.float32
    # print(noise_matrix[0][0])  # For the convenience of observation, only the first line element of the first channel is output here

    # 2. Adding to the original multidimensional matrix can achieve the effect of adding Gaussian noise
    matrix = matrix + noise_matrix

    # 3. Output matrix with added noise
    # print(">>>>>>>>>added gaussain noise with method 2")
    # print(matrix[0][0])  # For the convenience of observation, only the first line element of the first channel is output here

    return matrix

