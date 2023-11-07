import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from tqdm import trange


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


result_list = []
X = None
# 读取CSV文件
for i in trange(6000):
    csv_file_path = f'/home/r/DP_Data/悬浮-SF6/UHF/C1Trace{i:05}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data = df['Time'].values
    amplitude_data = df['Ampl'].values
    # lv bo
    cutoff_frequency = 3e8
    nyquist_frequency = 0.5 * 1.0 / (time_data[1] - time_data[0])
    cutoff_normalized = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(N=2, Wn=cutoff_normalized, btype='highpass')
    filtered_amplitude = signal.filtfilt(b, a, amplitude_data)
    amplitude_data = filtered_amplitude
    # 傅里叶变换
    sampling_rate = 1.0 / (time_data[1] - time_data[0])  # 计算采样率
    x, result = FFT(sampling_rate, amplitude_data)
    result_list.append(result)
    X = x
plt.rc('font', size=30)
plt.figure(figsize=(19, 9))
#plt.figure(figsize=(18, 12))
plt.plot(X, sum(result_list) / len(result_list))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([3e8, 1.5e9])
plt.tight_layout()
plt.show()
