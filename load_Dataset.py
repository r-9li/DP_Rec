import os.path as path
import random

import numpy as np
import torch
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm


class PD_Dataset(data.Dataset):
    def __init__(self, Dataset_Path, Data_Source, DataType, train_val_test, classes_num, Cache=False):
        self.Dataset_Path = path.join(Dataset_Path, Data_Source)

        assert train_val_test in ['train', 'val', 'test']

        self.label = {}
        self.Data_Name = []
        self.Data = {}
        self.cache = Cache
        self.Data_Filepath = {}
        self.DataName_RandomMix = {}

        self.train_val_test = train_val_test

        for i, datatype in enumerate(DataType):

            Final_Data_Path = path.join(self.Dataset_Path, datatype, train_val_test)
            with open(path.join(Final_Data_Path, 'Label.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    split_line = line.split(' ')
                    self.label[split_line[0]] = str(int(split_line[1]) + classes_num * i)
                    self.Data_Name.append(split_line[0])

        if self.cache:
            for filename in tqdm(self.Data_Name):
                filename += '.npy'
                if filename.startswith('UHF'):
                    file_root_path = path.join(self.Dataset_Path, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    file_root_path = path.join(self.Dataset_Path, 'UL', train_val_test)
                else:
                    file_root_path = None
                data = np.load(path.join(file_root_path, filename))
                self.Data[filename[:-4]] = data
                self.Data_Filepath[filename[:-4]] = path.join(file_root_path, filename)
        else:
            for filename in self.Data_Name:
                filename += '.npy'
                if filename.startswith('UHF'):
                    file_root_path = path.join(self.Dataset_Path, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    file_root_path = path.join(self.Dataset_Path, 'UL', train_val_test)
                else:
                    file_root_path = None
                self.Data_Filepath[filename[:-4]] = path.join(file_root_path, filename)

        for Class in list(set(list(self.label.values()))):
            self.DataName_RandomMix[Class] = {}
            for datatype in DataType:
                self.DataName_RandomMix[Class][datatype] = []

        for key, value in self.label.items():
            if key.startswith('UHF'):
                self.DataName_RandomMix[value]['UHF'].append(key)
            elif key.startswith('UL'):
                self.DataName_RandomMix[value]['UL'].append(key)

        self.lenght = len(self.Data_Name)
        print(self.lenght)
        assert self.lenght == len(self.Data_Filepath)
        assert self.lenght == len(self.label.values())

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        data_name = self.Data_Name[index]
        if self.cache:
            data = self.Data[data_name]
            label = self.label[data_name]
        else:
            data = np.load(self.Data_Filepath[data_name])
            label = self.label[data_name]

        if self.train_val_test == 'train':
            #  random scale augment
            if random.random() < 0.7:
                random_scale = random.uniform(0.5, 2.1)
                data = data * random_scale

            # random mix augment
            if random.random() < 0.45:
                if data_name.startswith('UHF'):
                    if len(self.DataName_RandomMix[label]['UHF']) != 0:  # Not Empty
                        random_choice_data_name = random.choice(self.DataName_RandomMix[label]['UHF'])
                        if self.cache:
                            random_choice_data = self.Data[random_choice_data_name]
                        else:
                            random_choice_data = np.load(self.Data_Filepath[random_choice_data_name])
                    else:
                        random_choice_data = data
                elif data_name.startswith('UL'):
                    if len(self.DataName_RandomMix[label]['UL']) != 0:  # Not Empty
                        random_choice_data_name = random.choice(self.DataName_RandomMix[label]['UL'])
                        if self.cache:
                            random_choice_data = self.Data[random_choice_data_name]
                        else:
                            random_choice_data = np.load(self.Data_Filepath[random_choice_data_name])
                    else:
                        random_choice_data = data
                else:
                    random_choice_data = None

                data = 0.8 * data + 0.2 * random_choice_data

        data = torch.from_numpy(data.astype(np.float32)).view(1, -1)
        label = np.array(int(label))
        label = torch.from_numpy(label.astype(np.int64))
        return data, label


class PD_PLDataModule(LightningDataModule):
    def __init__(self, Dataset_Path, Data_Source, DataType, Num_Workers, Pin_Memory, Batch_Size, Classes_num,
                 Cache=False):
        super().__init__()
        self.Dataset_Path = Dataset_Path
        self.Data_Source = Data_Source
        self.DataType = DataType
        self.Num_Workers = Num_Workers
        self.Pin_Memory = Pin_Memory
        self.batch_size = Batch_Size
        self.Classes_num = Classes_num
        self.Cache = Cache

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = PD_Dataset(self.Dataset_Path, self.Data_Source, self.DataType, 'train', self.Classes_num,
                                         self.Cache)
            self.val_data = PD_Dataset(self.Dataset_Path, self.Data_Source, self.DataType, 'val', self.Classes_num,
                                       self.Cache)
        if stage == 'test':
            self.test_data = PD_Dataset(self.Dataset_Path, self.Data_Source, self.DataType, 'test', self.Classes_num,
                                        self.Cache)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                          batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                          batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                          batch_size=self.batch_size, shuffle=False)
