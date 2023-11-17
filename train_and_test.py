import argparse

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from load_Dataset import PD_PLDataModule
from load_Dataset_CrossDomain import PD_PLDataModule_CrossDomain
from load_Dataset_CrossDomain import Set_Spurious_Label_Callback
from model import DP_Net_CrossDomain_Lightning
from model import DP_Net_Lightning
from model_lib.Conv1D_Encoder import Encoder1D_Param
from lightning.pytorch.loggers import TensorBoardLogger


def main(Opt):
    pl.seed_everything(42)
    logger = TensorBoardLogger("tb_logs", name="DP_model")

    if Opt.Cross_Domain:  # Cross Domain
        checkpoint_callback = ModelCheckpoint(dirpath='experience/models', monitor='val_final_acc_target',
                                              filename='{epoch}-{val_final_acc_target:.4f}',
                                              auto_insert_metric_name=True, save_weights_only=False, mode='max',
                                              save_top_k=100)
        earlystop_callback = EarlyStopping(monitor='val_final_acc_target', mode='max', patience=10)

        set_spurious_label_callback = Set_Spurious_Label_Callback()
        trainer = pl.Trainer(callbacks=[checkpoint_callback, earlystop_callback, set_spurious_label_callback],
                             benchmark=True,
                             max_epochs=3000,
                             accelerator='gpu', logger=logger,
                             devices='auto')  # strategy="ddp", sync_batchnorm=True   # remote
        PD_DataPL_CrossDomain = PD_PLDataModule_CrossDomain(Opt.Dataset_Path, Opt.Source_Domain, Opt.Target_Domain,
                                                            Opt.DataType, Opt.Num_Workers, Opt.Pin_Memory,
                                                            Opt.Batch_Size, Opt.Use_Spurious_Label, Opt.Dataset_Cache,
                                                            Opt.Target_Source_Rate,
                                                            Opt.Spurious_Label_Update, Opt.Init_Conf_Threshold)
        if Opt.train_test == 'train':
            DP_Network_CrossDomain = DP_Net_CrossDomain_Lightning(Opt.Class_Number, Opt.Input_Size, Opt.Encoder_Param,
                                                                  Opt.Domain_Loss_Weight, Opt.Diff_Loss_Weight,
                                                                  Opt.Target_Loss_Weight, Opt.Use_Spurious_Label,
                                                                  Opt.Drop_Rate,
                                                                  Opt.Original_Compatible, Opt.Hidden_Size,
                                                                  Opt.Discriminator_Hidden_Size, Opt.Target_Domain_Num,
                                                                  Opt.Acc_Check_Step)
            if Opt.Resume:
                print("Resume from ", Opt.CheckPoint)
                DP_Network_CrossDomain.load_state_dict(torch.load(Opt.CheckPoint)['state_dict'])
            else:
                DP_Network_CrossDomain.init_subnetwork(Opt.Init_Weight, Opt.Additional_Init)

            trainer.fit(model=DP_Network_CrossDomain, datamodule=PD_DataPL_CrossDomain)

        elif Opt.train_test == 'test':
            DP_Network_CrossDomain_Test = DP_Net_CrossDomain_Lightning(Opt.Class_Number, Opt.Input_Size,
                                                                       Opt.Encoder_Param, Opt.Domain_Loss_Weight,
                                                                       Opt.Diff_Loss_Weight, Opt.Target_Loss_Weight,
                                                                       Opt.Use_Spurious_Label,
                                                                       Opt.Drop_Rate,
                                                                       Opt.Original_Compatible, Opt.Hidden_Size,
                                                                       Opt.Discriminator_Hidden_Size,
                                                                       Opt.Target_Domain_Num,
                                                                       Opt.Acc_Check_Step)
            DP_Network_CrossDomain_Test.load_state_dict(torch.load(Opt.CheckPoint)['state_dict'])

            trainer.test(model=DP_Network_CrossDomain_Test, datamodule=PD_DataPL_CrossDomain)

    else:  # Normal
        checkpoint_callback = ModelCheckpoint(dirpath='experience/models', monitor='val_final_acc',
                                              filename='{epoch}-{val_final_acc:.4f}',
                                              auto_insert_metric_name=True, save_weights_only=False, mode='max',
                                              save_top_k=100)
        earlystop_callback = EarlyStopping(monitor='val_final_acc', mode='max', patience=6)

        trainer = pl.Trainer(callbacks=[checkpoint_callback, earlystop_callback], benchmark=True,
                             max_epochs=3000, logger=logger,
                             accelerator='gpu', devices='auto')
        PD_DataPL = PD_PLDataModule(Opt.Dataset_Path, Opt.Data_Source, Opt.DataType, Opt.Num_Workers, Opt.Pin_Memory,
                                    Opt.Batch_Size, Opt.Dataset_Cache)

        if Opt.train_test == 'train':
            DP_Network = DP_Net_Lightning(Opt.Class_Number, Opt.Input_Size, Opt.Encoder_Param,
                                          Opt.Drop_Rate,
                                          Opt.Original_Compatible, Opt.Hidden_Size, Opt.Acc_Check_Step)
            trainer.fit(model=DP_Network, datamodule=PD_DataPL)

        elif Opt.train_test == 'test':
            DP_Network_Test = DP_Net_Lightning(Opt.Class_Number, Opt.Input_Size, Opt.Encoder_Param,
                                               Opt.Drop_Rate,
                                               Opt.Original_Compatible, Opt.Hidden_Size, Opt.Acc_Check_Step)
            DP_Network_Test.load_state_dict(torch.load(Opt.CheckPoint)['state_dict'])

            trainer.test(model=DP_Network_Test, datamodule=PD_DataPL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_test', type=str, default='test')
    parser.add_argument('--Class_Number', type=int, default=4)
    parser.add_argument('--Input_Size', type=int, default=1)
    parser.add_argument('--Encoder_Param', type=dict, default=Encoder1D_Param)
    parser.add_argument('--Drop_Rate', type=float, default=0.1)
    parser.add_argument('--Original_Compatible', type=str, default="non-conservative")
    parser.add_argument('--Hidden_Size', type=int, default=128)
    parser.add_argument('--Acc_Check_Step', type=int, default=10)
    parser.add_argument('--Dataset_Path', type=str, default='/home/r/DP_Data/Processed_Data')  # remote
    parser.add_argument('--Data_Source', type=str, default='SF6')
    parser.add_argument('--DataType', type=list, default=['UHF', 'UL'])
    parser.add_argument('--Dataset_Cache', type=bool, default=True)
    parser.add_argument('--Num_Workers', type=int, default=16)  # remote
    parser.add_argument('--Pin_Memory', type=bool, default=False)  # remote
    parser.add_argument('--Batch_Size', type=int, default=16)
    parser.add_argument('--CheckPoint', type=str,
                        default='/mnt/c/Users/26593/Desktop/DP_Rec/experience/models/epoch=7-val_final_acc_target=50.4417.ckpt')
    parser.add_argument('--Resume', type=bool, default=False)

    parser.add_argument('--Cross_Domain', type=bool, default=True)
    parser.add_argument('--Init_Weight', type=str,
                        default='/mnt/c/Users/26593/Desktop/DP_Rec/experience/models/epoch=0-val_final_acc=99.4219.ckpt')  # remote   /home/r/DP/experience/models
    parser.add_argument('--Source_Domain', type=str, default='SF6')
    parser.add_argument('--Target_Domain', type=str, default='C4')
    parser.add_argument('--Target_Source_Rate', type=float, default=0.1)
    parser.add_argument('--Spurious_Label_Update', type=int, default=1)
    parser.add_argument('--Discriminator_Hidden_Size', type=int, default=128)
    parser.add_argument('--Target_Domain_Num', type=int, default=1)
    parser.add_argument('--Additional_Init', type=bool, default=False)
    parser.add_argument('--Domain_Loss_Weight', type=float, default=0.3)
    parser.add_argument('--Diff_Loss_Weight', type=float, default=0.2)
    parser.add_argument('--Target_Loss_Weight', type=float, default=0.15)
    parser.add_argument('--Init_Conf_Threshold', type=float, default=0.75)
    parser.add_argument('--Use_Spurious_Label', type=bool, default=False)

    opt = parser.parse_args()
    main(opt)
