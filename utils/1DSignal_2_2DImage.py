import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import trange


def signal_scaling(data):
    standardized_data = (data - np.mean(data)) / np.std(data)
    scaled_data = standardized_data / np.max(np.abs(standardized_data))
    return scaled_data


def STFT_result_scaling(data, scaling=1):
    magnitude = np.abs(data)
    log_magnitude = 10 * np.log10(magnitude + 1e-10)
    normalized_log_magnitude = (log_magnitude - np.mean(log_magnitude)) / np.std(log_magnitude)
    normalized_log_magnitude = (normalized_log_magnitude - np.min(normalized_log_magnitude)) / (
            np.max(normalized_log_magnitude) - np.min(normalized_log_magnitude))
    normalized_log_magnitude = normalized_log_magnitude * scaling
    return normalized_log_magnitude


for i in trange(10000):
    csv_file_path = f'/media/r/T7 Shield/尖端-C4/jian-C4-UHF/C1Trace{i:05d}.csv'  # 将文件路径替换为实际文件路径
    df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
    df.columns = ['Time', 'Ampl']

    # 提取时间和幅值数据
    time_data = np.array(df['Time'].values)
    amplitude_data = np.array(df['Ampl'].values)

    fs = 1.0 / (time_data[1] - time_data[0])  # 采样频率
    data = amplitude_data  # 一维数据
    data = signal_scaling(data)

    f, t, nd = signal.stft(data, fs=fs, window='hann', nperseg=200, noverlap=None, nfft=None,
                           detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)

    nd = STFT_result_scaling(nd, scaling=200)
    np.save(f'/media/r/T7 Shield/jianduan-C4/jian-C4-UHF/{i:05d}.npy', nd)
