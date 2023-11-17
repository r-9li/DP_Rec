import os.path as path
import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
from load_Dataset import PD_Dataset


class Set_Spurious_Label_Callback(Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % trainer.train_dataloader.dataset.spurious_label_update == 0:
            trainer.train_dataloader.dataset.Set_Spurious_Label(pl_module)


class PD_Dataset_CrossDomain(data.Dataset):
    def __init__(self, Dataset_Path, Source_Domain, Target_Domain, DataType, train_val_test, Use_Spurious_Label,
                 Cache=False,
                 Target_Source_rate=0.1, Spurious_Label_Update=5, Init_conf_threshold=0.75):
        self.Use_Spurious_Label = Use_Spurious_Label
        self.Dataset_Path_Source = path.join(Dataset_Path, Source_Domain)
        self.Dataset_Path_Target = path.join(Dataset_Path, Target_Domain)

        assert train_val_test in ['train', 'val', 'test']

        self.cache = Cache
        self.train_val_test = train_val_test
        self.spurious_label_update = Spurious_Label_Update

        # Source Domain
        self.Source_Label = {}
        self.Source_Data_Name = []  # 文件名
        self.Source_Data = {}
        self.Source_Data_Filepath = {}  # 文件名对应的文件路径
        self.Source_DataName_RandomMix = {}  # 数据增强用

        # Target Domain
        self.Target_Label = {}  # 伪标签
        self.Target_Data_Name = []
        self.Target_Data_Name_nosp = {}  # 分来源储存，直接从这里面抽取
        self.Target_Data = {}
        self.Target_Data_Filepath = {}
        self.Target_DataName_RandomMix = {}
        self.Target_Label_conf = {}  # 置信度

        self.Conf_threshold = Init_conf_threshold

        for datatype in DataType:
            if not self.Use_Spurious_Label:
                self.Target_Data_Name_nosp[datatype] = []
            # Load Source Domain
            Source_Final_Data_Path = path.join(self.Dataset_Path_Source, datatype, train_val_test)
            with open(path.join(Source_Final_Data_Path, 'Label.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    split_line = line.split(' ')
                    self.Source_Label[split_line[0]] = split_line[1]
                    self.Source_Data_Name.append(split_line[0])

            # Load Target Domain
            Target_Final_Data_Path = path.join(self.Dataset_Path_Target, datatype, train_val_test)
            with open(path.join(Target_Final_Data_Path, 'Label.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    split_line = line.split(' ')
                    if random.random() < Target_Source_rate:  # 按比例采样
                        if self.Use_Spurious_Label:
                            self.Target_Data_Name.append(split_line[0])
                        else:
                            self.Target_Data_Name.append(split_line[0])
                            self.Target_Data_Name_nosp[datatype].append(split_line[0])

        # Load Source Domain
        if self.cache:
            for filename in tqdm(self.Source_Data_Name):
                filename += '.npy'
                if filename.startswith('UHF'):
                    source_file_root_path = path.join(self.Dataset_Path_Source, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    source_file_root_path = path.join(self.Dataset_Path_Source, 'UL', train_val_test)
                else:
                    source_file_root_path = None
                data = np.load(path.join(source_file_root_path, filename))
                self.Source_Data[filename[:-4]] = data
                self.Source_Data_Filepath[filename[:-4]] = path.join(source_file_root_path, filename)
        else:
            for filename in self.Source_Data_Name:
                filename += '.npy'
                if filename.startswith('UHF'):
                    source_file_root_path = path.join(self.Dataset_Path_Source, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    source_file_root_path = path.join(self.Dataset_Path_Source, 'UL', train_val_test)
                else:
                    source_file_root_path = None
                self.Source_Data_Filepath[filename[:-4]] = path.join(source_file_root_path, filename)

        # Load Target Domain
        if self.cache:
            for filename in tqdm(self.Target_Data_Name):
                filename += '.npy'
                if filename.startswith('UHF'):
                    target_file_root_path = path.join(self.Dataset_Path_Target, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    target_file_root_path = path.join(self.Dataset_Path_Target, 'UL', train_val_test)
                else:
                    target_file_root_path = None
                data = np.load(path.join(target_file_root_path, filename))
                self.Target_Data[filename[:-4]] = data
                self.Target_Data_Filepath[filename[:-4]] = path.join(target_file_root_path, filename)
        else:
            for filename in self.Target_Data_Name:
                filename += '.npy'
                if filename.startswith('UHF'):
                    target_file_root_path = path.join(self.Dataset_Path_Target, 'UHF', train_val_test)
                elif filename.startswith('UL'):
                    target_file_root_path = path.join(self.Dataset_Path_Target, 'UL', train_val_test)
                else:
                    target_file_root_path = None
                self.Target_Data_Filepath[filename[:-4]] = path.join(target_file_root_path, filename)

        # Source Domain Mix
        for Class in list(set(list(self.Source_Label.values()))):
            self.Source_DataName_RandomMix[Class] = {}
            self.Target_DataName_RandomMix[Class] = {}
            for datatype in DataType:
                self.Source_DataName_RandomMix[Class][datatype] = []
                self.Target_DataName_RandomMix[Class][datatype] = []

        for key, value in self.Source_Label.items():
            if key.startswith('UHF'):
                self.Source_DataName_RandomMix[value]['UHF'].append(key)
            elif key.startswith('UL'):
                self.Source_DataName_RandomMix[value]['UL'].append(key)

        self.lenght = len(self.Source_Data_Name)
        print('Source Domain Number', self.lenght)
        assert self.lenght == len(self.Source_Data_Filepath)
        assert self.lenght == len(self.Source_Label.values())

        print("Target Domain Number", len(self.Target_Data_Name))
        assert len(self.Target_Data_Name) == len(self.Target_Data_Filepath)

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        data_name = self.Source_Data_Name[index]

        # Get Source Domain Data
        if self.cache:
            data = self.Source_Data[data_name]
            label = self.Source_Label[data_name]
        else:
            data = np.load(self.Source_Data_Filepath[data_name])
            label = self.Source_Label[data_name]

        if self.train_val_test == 'train':
            #  random scale augment
            if random.random() < 0.7:
                random_scale = random.uniform(0.5, 2.1)
                data = data * random_scale

            # random mix augment
            if random.random() < 0.45:
                if data_name.startswith('UHF'):
                    if len(self.Source_DataName_RandomMix[label]['UHF']) != 0:  # Not Empty
                        random_choice_data_name = random.choice(self.Source_DataName_RandomMix[label]['UHF'])
                        if self.cache:
                            random_choice_data = self.Source_Data[random_choice_data_name]
                        else:
                            random_choice_data = np.load(self.Source_Data_Filepath[random_choice_data_name])
                    else:
                        random_choice_data = data
                elif data_name.startswith('UL'):
                    if len(self.Source_DataName_RandomMix[label]['UL']) != 0:  # Not Empty
                        random_choice_data_name = random.choice(self.Source_DataName_RandomMix[label]['UL'])
                        if self.cache:
                            random_choice_data = self.Source_Data[random_choice_data_name]
                        else:
                            random_choice_data = np.load(self.Source_Data_Filepath[random_choice_data_name])
                    else:
                        random_choice_data = data
                else:
                    random_choice_data = None

                data = 0.8 * data + 0.2 * random_choice_data

        #  Get Target Domain Data
        if self.Use_Spurious_Label:
            if data_name.startswith('UHF'):
                if len(self.Target_DataName_RandomMix[label]['UHF']) != 0:  # Not Empty
                    random_choice_data_name = random.choice(self.Target_DataName_RandomMix[label]['UHF'])
                    if self.cache:
                        target_data = self.Target_Data[random_choice_data_name]
                        conf = self.Target_Label_conf[random_choice_data_name]
                    else:
                        target_data = np.load(self.Target_Data_Filepath[random_choice_data_name])
                        conf = self.Target_Label_conf[random_choice_data_name]
                else:
                    target_data = data
                    conf = 0.
            elif data_name.startswith('UL'):
                if len(self.Target_DataName_RandomMix[label]['UL']) != 0:  # Not Empty
                    random_choice_data_name = random.choice(self.Target_DataName_RandomMix[label]['UL'])
                    if self.cache:
                        target_data = self.Target_Data[random_choice_data_name]
                        conf = self.Target_Label_conf[random_choice_data_name]
                    else:
                        target_data = np.load(self.Target_Data_Filepath[random_choice_data_name])
                        conf = self.Target_Label_conf[random_choice_data_name]
                else:
                    target_data = data
                    conf = 0.
            else:
                target_data = None
                conf = 0.
        else:
            if data_name.startswith('UHF'):
                random_choice_data_name = random.choice(self.Target_Data_Name_nosp['UHF'])
                target_data = self.Target_Data[random_choice_data_name]
            elif data_name.startswith('UL'):
                random_choice_data_name = random.choice(self.Target_Data_Name_nosp['UL'])
                target_data = self.Target_Data[random_choice_data_name]

        if self.train_val_test == 'train':
            #  random scale augment
            if random.random() < 0.7:
                random_scale = random.uniform(0.5, 2.1)
                target_data = target_data * random_scale

            # random mix augment
            if random.random() < 0.45:
                if self.Use_Spurious_Label:
                    if data_name.startswith('UHF'):
                        if len(self.Target_DataName_RandomMix[label]['UHF']) != 0:  # Not Empty
                            random_choice_data_name = random.choice(self.Target_DataName_RandomMix[label]['UHF'])
                            if self.cache:
                                random_choice_data = self.Target_Data[random_choice_data_name]
                            else:
                                random_choice_data = np.load(self.Target_Data_Filepath[random_choice_data_name])
                        else:
                            random_choice_data = target_data
                    elif data_name.startswith('UL'):
                        if len(self.Target_DataName_RandomMix[label]['UL']) != 0:  # Not Empty
                            random_choice_data_name = random.choice(self.Target_DataName_RandomMix[label]['UL'])
                            if self.cache:
                                random_choice_data = self.Target_Data[random_choice_data_name]
                            else:
                                random_choice_data = np.load(self.Target_Data_Filepath[random_choice_data_name])
                        else:
                            random_choice_data = target_data
                    else:
                        random_choice_data = None
                else:
                    if data_name.startswith('UHF'):
                        random_choice_data_name = random.choice(self.Target_Data_Name_nosp['UHF'])
                        random_choice_data = self.Target_Data[random_choice_data_name]
                    elif data_name.startswith('UL'):
                        random_choice_data_name = random.choice(self.Target_Data_Name_nosp['UL'])
                        random_choice_data = self.Target_Data[random_choice_data_name]

                target_data = 0.8 * target_data + 0.2 * random_choice_data

        data = torch.from_numpy(data.astype(np.float32)).view(1, -1)
        label = np.array(int(label))
        label = torch.from_numpy(label.astype(np.int64))
        target_data = torch.from_numpy(target_data.astype(np.float32)).view(1, -1)

        # conf to weight
        if self.Use_Spurious_Label:
            scale = 10.
            conf = (conf - self.Conf_threshold) * scale / (1 - self.Conf_threshold)
            target_loss_weight = F.sigmoid(torch.tensor(conf - scale / 2))
        else:
            target_loss_weight = 9  # 禁止返回None

        return (data, target_data, label, target_loss_weight)

    def Set_Spurious_Label(self, pl_model):
        if self.Use_Spurious_Label:
            self.Target_Label.clear()
            model: nn.Module = pl_model.network
            model.eval()
            with torch.no_grad():
                for data_name in tqdm(self.Target_Data_Name):
                    data = self.Target_Data[data_name]
                    data = torch.from_numpy(data.astype(np.float32)).view(1, -1)
                    data = data.unsqueeze(0)
                    data = data.cuda()
                    pred = model(data)
                    pred = F.softmax(pred, dim=1)
                    conf, pred = torch.topk(pred, k=1, dim=1, largest=True, sorted=True)
                    pred = pred.squeeze()
                    if conf > self.Conf_threshold:
                        self.Target_Label[data_name] = str(int(pred))
                        self.Target_Label_conf[data_name] = float(conf)

            # Target Domain Mix
            for key, value in self.Target_DataName_RandomMix.items():
                for key1, value1 in value.items():
                    value1.clear()

            for key, value in self.Target_Label.items():
                if key.startswith('UHF'):
                    self.Target_DataName_RandomMix[value]['UHF'].append(key)
                elif key.startswith('UL'):
                    self.Target_DataName_RandomMix[value]['UL'].append(key)

            print('Rate:', len(self.Target_Label) / len(self.Target_Data_Name))
            self.Conf_threshold += 0.05
            self.Conf_threshold = min(0.95, self.Conf_threshold)
            model.train()


class PD_PLDataModule_CrossDomain(LightningDataModule):
    def __init__(self, Dataset_Path, Source_Domain, Target_Domain, DataType, Num_Workers, Pin_Memory, Batch_Size,
                 Use_Spurious_Label,
                 Cache=False, Target_Source_rate=0.1,
                 Spurious_Label_Update=5, Init_conf_threshold=0.75):
        super().__init__()
        self.Dataset_Path = Dataset_Path
        self.Source_Domain = Source_Domain
        self.Target_Domain = Target_Domain
        self.DataType = DataType
        self.Num_Workers = Num_Workers
        self.Pin_Memory = Pin_Memory
        self.batch_size = Batch_Size
        self.Use_Spurious_Label = Use_Spurious_Label
        self.Cache = Cache
        self.Target_Source_rate = Target_Source_rate
        self.Spurious_Label_Update = Spurious_Label_Update
        self.Init_conf_threshold = Init_conf_threshold

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = PD_Dataset_CrossDomain(self.Dataset_Path, self.Source_Domain, self.Target_Domain,
                                                     self.DataType, 'train', self.Use_Spurious_Label, self.Cache,
                                                     self.Target_Source_rate,
                                                     self.Spurious_Label_Update, self.Init_conf_threshold)

            self.val_data_source = PD_Dataset(self.Dataset_Path, self.Source_Domain, self.DataType, 'val', self.Cache)
            self.val_data_target = PD_Dataset(self.Dataset_Path, self.Target_Domain, self.DataType, 'val', self.Cache)
        if stage == 'test':
            self.test_data_source = PD_Dataset(self.Dataset_Path, self.Source_Domain, self.DataType, 'test', self.Cache)
            self.test_data_target = PD_Dataset(self.Dataset_Path, self.Target_Domain, self.DataType, 'test', self.Cache)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                          batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return [DataLoader(self.val_data_source, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                           batch_size=self.batch_size, shuffle=False),
                DataLoader(self.val_data_target, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                           batch_size=self.batch_size, shuffle=False)]

    def test_dataloader(self):
        return [DataLoader(self.test_data_source, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                           batch_size=self.batch_size, shuffle=False),
                DataLoader(self.test_data_target, num_workers=self.Num_Workers, pin_memory=self.Pin_Memory,
                           batch_size=self.batch_size, shuffle=False)]
