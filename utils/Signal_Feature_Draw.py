import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pandas as pd
from scipy import stats

Data_1 = []
Data_2 = []

Data_Type_1 = 'Gravity Frequcenty'  # x轴
Data_Type_2 = 'Average Frequency'  # y轴
assert Data_Type_1 in ['Skewness', 'Kurtosis', 'Gravity Frequcenty', 'Average Frequency',
                       'Frequency Standard Deviation', 'RMS Frequency']
assert Data_Type_2 in ['Skewness', 'Kurtosis', 'Gravity Frequcenty', 'Average Frequency',
                       'Frequency Standard Deviation', 'RMS Frequency']
Data_1_1 = None  # 数据来源_类型
Data_1_2 = None
Data_2_1 = None
Data_2_2 = None


def seg_mean(A, num):
    return np.mean(A.reshape(-1, num), axis=1)


def get_time_domain_feature(data):
    """
    提取 15个 时域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @return: shape 为 (m, 15)  的 2D array 数据，其中，m 为样本个数。即 每个样本的16个时域特征
    """
    rows, cols = data.shape

    # 有量纲统计量
    max_value = np.amax(data, axis=1)  # 最大值
    peak_value = np.amax(abs(data), axis=1)  # 最大绝对值
    min_value = np.amin(data, axis=1)  # 最小值
    mean = np.mean(data, axis=1)  # 均值
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=1)  # 绝对平均值
    rms = np.sqrt(np.sum(data ** 2, axis=1) / cols)  # 均方根值
    square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=1) / cols) ** 2  # 方根幅值
    # variance = np.var(data, axis=1)  # 方差
    std = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=1)  # 峭度
    skewness = stats.skew(data, axis=1)  # 偏度
    # mean_amplitude = np.sum(np.abs(data), axis=1) / cols  # 平均幅值 == 绝对平均值

    # 无量纲统计量
    clearance_factor = peak_value / square_root_amplitude  # 裕度指标
    shape_factor = rms / abs_mean  # 波形指标
    impulse_factor = peak_value / abs_mean  # 脉冲指标
    crest_factor = peak_value / rms  # 峰值指标
    # kurtosis_factor = kurtosis / (rms**4)  # 峭度指标

    features = [max_value, peak_value, min_value, mean, p_p_value, abs_mean, rms, square_root_amplitude,
                std, kurtosis, skewness, clearance_factor, shape_factor, impulse_factor, crest_factor]

    return features


def get_frequency_domain_feature(data, sampling_frequency):
    """
    提取 4个 频域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @param sampling_frequency: 采样频率
    @return: shape 为 (m, 4)  的 2D array 数据，其中，m 为样本个数。即 每个样本的4个频域特征
    """
    data_fft = np.fft.fft(data, axis=1)
    m, N = data_fft.shape  # 样本个数 和 信号长度

    # 傅里叶变换是对称的，只需取前半部分数据，否则由于 频率序列 是 正负对称的，会导致计算 重心频率求和 等时正负抵消
    mag = np.abs(data_fft)[:, : N // 2]  # 信号幅值
    freq = np.fft.fftfreq(N, 1 / sampling_frequency)[: N // 2]

    ps = mag ** 2 / N  # 功率谱

    fc = np.sum(freq * ps, axis=1) / np.sum(ps, axis=1)  # 重心频率
    mf = np.mean(ps, axis=1)  # 平均频率
    rmsf = np.sqrt(np.sum(ps * np.square(freq), axis=1) / np.sum(ps, axis=1))  # 均方根频率

    freq_tile = np.tile(freq.reshape(1, -1), (m, 1))  # 复制 m 行
    fc_tile = np.tile(fc.reshape(-1, 1), (1, freq_tile.shape[1]))  # 复制 列，与 freq_tile 的列数对应
    vf = np.sqrt(np.sum(np.square(freq_tile - fc_tile) * ps, axis=1) / np.sum(ps, axis=1))  # 频率方差

    features = [fc, mf, rmsf, vf]

    return features


for i in trange(10000):
    csv_file_path = f'/home/r/DP_Data/尖端-C4/UL/C2Trace{i:05}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data_1 = df['Time'].values
    amplitude_data_1 = df['Ampl'].values

    csv_file_path = f'/home/r/DP_Data/尖端-SF6/UL/C2Trace{i:05}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data_2 = df['Time'].values
    amplitude_data_2 = df['Ampl'].values

    sampling_rate = 1.0 / (time_data_1[1] - time_data_1[0])

    Data_1.append(amplitude_data_1)
    Data_2.append(amplitude_data_2)

Data_1 = np.array(Data_1)
Data_2 = np.array(Data_2)

Time_Domain_1 = get_time_domain_feature(Data_1)
Time_Domain_2 = get_time_domain_feature(Data_2)

Frequency_Domain_1 = get_frequency_domain_feature(Data_1, sampling_frequency=sampling_rate)
Frequency_Domain_2 = get_frequency_domain_feature(Data_2, sampling_frequency=sampling_rate)

if Data_Type_1 == 'Skewness':
    Data_1_1 = Time_Domain_1[10]
    Data_2_1 = Time_Domain_2[10]
elif Data_Type_1 == 'Kurtosis':
    Data_1_1 = Time_Domain_1[9]
    Data_2_1 = Time_Domain_2[9]
elif Data_Type_1 == 'Gravity Frequcenty':
    Data_1_1 = Frequency_Domain_1[0]
    Data_2_1 = Frequency_Domain_2[0]
elif Data_Type_1 == 'Average Frequency':
    Data_1_1 = Frequency_Domain_1[1]
    Data_2_1 = Frequency_Domain_2[1]
elif Data_Type_1 == 'Frequency Standard Deviation':
    Data_1_1 = Frequency_Domain_1[3]
    Data_2_1 = Frequency_Domain_2[3]
elif Data_Type_1 == 'RMS Frequency':
    Data_1_1 = Frequency_Domain_1[2]
    Data_2_1 = Frequency_Domain_2[2]

if Data_Type_2 == 'Skewness':
    Data_1_2 = Time_Domain_1[10]
    Data_2_2 = Time_Domain_2[10]
elif Data_Type_2 == 'Kurtosis':
    Data_1_2 = Time_Domain_1[9]
    Data_2_2 = Time_Domain_2[9]
elif Data_Type_2 == 'Gravity Frequcenty':
    Data_1_2 = Frequency_Domain_1[0]
    Data_2_2 = Frequency_Domain_2[0]
elif Data_Type_2 == 'Average Frequency':
    Data_1_2 = Frequency_Domain_1[1]
    Data_2_2 = Frequency_Domain_2[1]
elif Data_Type_2 == 'Frequency Standard Deviation':
    Data_1_2 = Frequency_Domain_1[3]
    Data_2_2 = Frequency_Domain_2[3]
elif Data_Type_2 == 'RMS Frequency':
    Data_1_2 = Frequency_Domain_1[2]
    Data_2_2 = Frequency_Domain_2[2]

# 生成一些示例数据
x1 = seg_mean(Data_1_1, 1000)
y1 = seg_mean(Data_1_2, 1000)

x2 = seg_mean(Data_2_1, 1000)
y2 = seg_mean(Data_2_2, 1000)
plt.rc('font', size=14)
# 创建图像和坐标轴
fig, ax = plt.subplots()

# 添加数据点
ax.scatter(x1, y1, c='red', marker='o', label='C4F7N/CO2/O2')
ax.scatter(x2, y2, c='blue', marker='s', label='SF6')

# 添加图例
ax.legend(loc='upper right')

# 显示图像
# plt.xlabel('t_r/ns')
# plt.ylabel('t_f/ns')
plt.title(f'{Data_Type_1}-{Data_Type_2}')
#plt.grid(True)
plt.tight_layout()
plt.show()
