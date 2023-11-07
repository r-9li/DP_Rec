import numpy as np
import pandas as pd
from scipy.fftpack import fft
from tqdm import trange
import random
import os.path as path


def Result_Scaling(data, scaling=1):
    standardized_data = (data - np.mean(data)) / np.std(data)
    min_max_scaled_data = (standardized_data - np.min(standardized_data)) / (
            np.max(standardized_data) - np.min(standardized_data))
    scaled_data = min_max_scaled_data * scaling
    return scaled_data


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


Data_Source = 'SF6'
Data_Type = 'UHF'
Root_Folder = '/home/r/DP_Data'
Process_Type = 'Frequency'  # Time or Frequency
Data_Folder = [Root_Folder + '/尖端-' + Data_Source + '/' + Data_Type,
               Root_Folder + '/悬浮-' + Data_Source + '/' + Data_Type,
               Root_Folder + '/微粒-' + Data_Source + '/' + Data_Type]  # Root_Folder+'/尖端-'+Data_Source+'/'+Data_Type
DataSave_Folder = Root_Folder + '/Processed_Data/' + Data_Source + '/' + Data_Type

f_train = open(path.join(DataSave_Folder, 'train', 'Label.txt'), 'w')
f_val = open(path.join(DataSave_Folder, 'val', 'Label.txt'), 'w')
f_test = open(path.join(DataSave_Folder, 'test', 'Label.txt'), 'w')

for i, Folder in enumerate(Data_Folder):
    for j in trange(10000):
        DataSave_Folder_Loop = DataSave_Folder
        flag_num = random.random()
        if flag_num <= 0.6:
            DataSave_Folder_Loop = path.join(DataSave_Folder_Loop, 'train')
            f = f_train
        elif 0.6 < flag_num <= 0.8:
            DataSave_Folder_Loop = path.join(DataSave_Folder_Loop, 'val')
            f = f_val
        elif flag_num > 0.8:
            DataSave_Folder_Loop = path.join(DataSave_Folder_Loop, 'test')
            f = f_test
        else:
            f = None

        if Data_Type == 'UHF':
            csv_file_path = Folder + f'/C1Trace{j:05d}.csv'
        elif Data_Type == 'UL':
            csv_file_path = Folder + f'/C2Trace{j:05d}.csv'
        else:
            csv_file_path = None

        df = pd.read_csv(csv_file_path, skiprows=4, header=1, sep=',')
        df.columns = ['Time', 'Ampl']

        time_data = np.array(df['Time'].values)
        amplitude_data = np.array(df['Ampl'].values)

        if Process_Type == 'Frequency':
            sampling_rate = 1.0 / (time_data[1] - time_data[0])  # 计算采样率
            x, result = FFT(sampling_rate, amplitude_data)
            Data = result
        else:
            Data = amplitude_data

        Data = Result_Scaling(Data, scaling=210)
        np.save(path.join(DataSave_Folder_Loop, Data_Type + f'{j + 10000 * i:06d}.npy'), Data)
        f.write(Data_Type + f'{j + 10000 * i:06d} {i:01d}' + '\n')

f_train.close()
f_val.close()
f_test.close()
