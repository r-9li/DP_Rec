import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial.distance import euclidean
from scipy import signal
from scipy.fftpack import fft, ifft

choice = 'moment'


def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result


def signal_scaling(data):
    standardized_data = (data - np.mean(data)) / np.std(data)
    scaled_data = standardized_data / np.max(np.abs(standardized_data))
    return scaled_data


def central_moment(data, k):
    mean = np.mean(data)
    return np.mean([(x - mean) ** k for x in data])


def moments(signal, order, normalized=True):
    """
    Calculate the moments for a 1D signal.

    Parameters:
        signal (array-like): Input 1D signal or data.
        order (int): Order of the moment to compute.
        normalized (bool): Whether to return the normalized moment (divided by the variance^order).

    Returns:
        float: The calculated moment of the specified order.
    """

    mu = np.mean(signal)
    central_moment = np.mean((signal - mu) ** order)

    if normalized and order > 1:
        variance = np.var(signal)
        return central_moment / (variance ** (order / 2))
    else:
        return central_moment


A = np.zeros((10, 10000))
B = np.zeros((10, 10000))
for i in trange(10000):
    csv_file_path = f'/media/r/T7 Shield/微粒-C4/UHF/C1Trace{i:05d}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data = np.array(df['Time'].values)
    amplitude_data = np.array(df['Ampl'].values)
    #amplitude_data = signal_scaling(amplitude_data)

    sampling_rate = 1.0 / (time_data[1] - time_data[0])  # 计算采样率
    x, result = FFT(sampling_rate, amplitude_data)
    data = result
    # fs = 1.0 / (time_data[1] - time_data[0])  # 采样频率
    # data = amplitude_data*10000  # 一维数据
    data = signal_scaling(data)
    for j in range(10):
        if choice == 'moment':
            A[j, i] = moments(data, j + 1, normalized=True)
        elif choice == 'central':
            A[j, i] = central_moment(data, j + 1)
        else:
            print('ERROR!')

for i in trange(10000):
    csv_file_path = f'/media/r/T7 Shield/微粒-SF6/wei-SF6-UHF/C1Trace{i:05d}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data = np.array(df['Time'].values)
    amplitude_data = np.array(df['Ampl'].values)
    #amplitude_data = signal_scaling(amplitude_data)

    sampling_rate = 1.0 / (time_data[1] - time_data[0])  # 计算采样率
    x, result = FFT(sampling_rate, amplitude_data)
    data = result
    # fs = 1.0 / (time_data[1] - time_data[0])  # 采样频率
    # data = amplitude_data * 10000  # 一维数据
    data = signal_scaling(data)
    for j in range(10):
        if choice == 'moment':
            B[j, i] = moments(data, j + 1, normalized=True)
        elif choice == 'central':
            B[j, i] = central_moment(data, j + 1)
        else:
            print('ERROR!')

# 创建模拟数据

# A_normalized = (A - np.mean(A, axis=1).reshape(-1, 1)) / np.std(A, axis=1).reshape(-1, 1)
# B_normalized = (B - np.mean(B, axis=1).reshape(-1, 1)) / np.std(B, axis=1).reshape(-1, 1)
# A_normalized[np.isnan(A_normalized)]=A[np.isnan(A_normalized)]
# B_normalized[np.isnan(B_normalized)]=B[np.isnan(B_normalized)]
#
# # 特征差异性分析
# _, p_values = ttest_ind(A_normalized.T, B_normalized.T, axis=0)
# effect_sizes = np.abs(np.mean(A_normalized, axis=1) - np.mean(B_normalized, axis=1)) / np.sqrt(
#     (np.var(A_normalized, axis=1) + np.var(B_normalized, axis=1)) / 2)
#
#
# # 稳定性分析
# def bhattacharyya_distance(d1, d2):
#     """计算Bhattacharyya距离"""
#     return -np.log(np.sum(np.sqrt(d1 * d2)))
#
#
# overlaps = []
# for i in range(A_normalized.shape[0]):
#     hist_A, _ = np.histogram(A_normalized[i, :], bins=100, density=True)
#     hist_B, _ = np.histogram(B_normalized[i, :], bins=100, density=True)
#     overlaps.append(bhattacharyya_distance(hist_A, hist_B))
#
# # 综合得分
# composite_scores = (1 - p_values) * effect_sizes * overlaps
#
# # 特征选择
# max_index = np.argmax(composite_scores)
# print(
#     f"The most distinguishing and stable feature is feature {max_index + 1} with a composite score of {composite_scores[max_index]:.4f}.")
# 计算每个数据的第二维的平均值和方差
mean_A = np.mean(A, axis=1)
variance_A = np.var(A, axis=1)
mean_B = np.mean(B, axis=1)
variance_B = np.var(B, axis=1)

print(np.abs((mean_A - mean_B) / (np.abs(mean_A) + np.abs(mean_B))),
      (np.abs(variance_A) + np.abs(variance_B)) / (np.abs(mean_A) + np.abs(mean_B)))
# 画图
fig, ax = plt.subplots()
x = 8
# 数据A的点与阴影
ax.scatter(range(1, x), mean_A[:x - 1], s=variance_A[:x - 1] * 10, alpha=0.5)
ax.plot(range(1, x), mean_A[:x - 1], color='blue', alpha=0.5)

# 数据B的点与阴影
ax.scatter(range(1, x), mean_B[:x - 1], s=variance_B[:x - 1] * 10, alpha=0.5, color='red')
ax.plot(range(1, x), mean_B[:x - 1], color='red', alpha=0.5)

ax.set_xlabel('Data Dimension')
ax.set_ylabel('Mean Value')
ax.legend()
plt.show()
