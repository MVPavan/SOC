# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:17:40 2019

@author: H242998
"""

import scipy.io
import pandas as pd
import numpy as np
import pickle
from pandas import ExcelWriter
from pathlib import Path

#from torch.utils.data import Dataset
import os
import re

pkl_file = './DATA/soc_db_v4.pkl'
###############################################################################

###############################################################################

def dt_ms_converter(x):
    x=re.split(r'(:|,|\s)\s*', x)
    return int(x[-1])+1000*(int(x[-3])+60*(int(x[-5])+60*int(x[-7])))

class SOCLoader():
    def __init__(self,):
        self.data_folder = './DATA/Discharging/'
        self.xls_path_all = Path(self.data_folder)/"consolidated.xlsx"
        self.xls_path_db = Path(self.data_folder)/"consolidated_db.xlsx"
        self.pkl_file = pkl_file
        self.average_factor = 400
        self.raw_files = {
            'train': {
                'cycle2_15A':"Data 8199 2391 3_5_2020 21_49_40_discharge15A_cycle2_SOC.xlsx",
                'cycle3_10A': 'Data 8199 2391 3_6_2020 15_57_41_Discharge_10A_cycle3_SOC.xlsx',
                'cycle4_13A': 'Data 8199 2391 3_9_2020 19_58_07_discharging13A_cycle4_SOC.xlsx',
                'cycle5_18A': 'Data 8199 2391 3_10_2020 17_08_23_discharge18A_cycle5_SOC.xlsx'
            },
            'test': {
                'cycle7_11A': 'Data 8199 2391 3_17_2020 13_21_37_discharging_11A_cycle7_SOC.xlsx',
            }
        }

    def process(self, excel_file):
        df = pd.read_excel(excel_file)
        df["dt_ms"] = df["Time"].apply(dt_ms_converter).diff().fillna(0)
        df['v'] = df[[col for col in df.columns if "VDC" in col]].min(axis = 1)
        df["i"] = df[[col for col in df.columns if "ADC" in col]]
        df["ah"] = (df["i"]*df["dt_ms"]/(3600*1000)).cumsum()
        df['temp'] = df[[col for col in df.columns if "<t" in col.lower()]].mean(axis = 1)
        df['soc'] = (df["ah"].max()-df["ah"])/df["ah"].max()                   # (4.2+ah)/4.2
        df['av'] = df['v'].rolling(min_periods=1, window=self.average_factor).mean()
        df['ai'] = df['i'].rolling(min_periods=1, window=self.average_factor).mean()
        return df, df[["v","i","dt_ms","ah","temp","soc","av","ai"]]

    def CreateDB(self,):
        _soc_db, soc_db = {},{}
        for key in self.raw_files.keys():
            _soc_db[key], soc_db[key] = {},{}
            for dataset in self.raw_files[key].keys():
                excel_file = '{}{}'.format(
                    self.data_folder, self.raw_files[key][dataset])
                # print(excel_file)
                _soc_db[key][dataset],soc_db[key][dataset] = self.process(excel_file)
                print("Processing complete from dataset: {}".format(dataset))
        self.save_xls([_soc_db, soc_db])
        with open(self.pkl_file, 'wb+') as fp:
            pickle.dump(soc_db, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
        

        
    def save_xls(self, db_list):
        for xls_path, dba in zip([self.xls_path_all,self.xls_path_db],db_list):
            with ExcelWriter(xls_path.as_posix()) as writer:
                for _, db in dba.items():
                    for key, df in db.items():
                        df.to_excel(writer,key)
                    writer.save()

###############################################################################
def PickleDB():
    data_processor = SOCLoader()
    data_processor.CreateDB()

# class SOCDataset(Dataset):


class SOCDataset():
    def __init__(self, soc_db, batch_len):
        self.soc_db = soc_db
        self.samples = []
        self._init_dataset()
        if batch_len > 1:
            self.length_check(batch_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def length_check(self, batch_len):
        redn = len(self.samples) % batch_len
        if redn >= 1:
            del self.samples[-redn:]

    def _init_dataset(self):
        for key in self.soc_db.keys():
            subset = self.soc_db[key]
            for i,row in subset.iterrows():
                self.samples.append((np.array(
                    [row["v"], row["temp"], row["av"], row["ai"]]), row["soc"]))
            print("Loading complete from dataset: {}".format(key))


def GetSOCdata(batch_len=1, pkl=False):
    if pkl:
        print("Calculating Average Current, Voltage and SOC .... ")
        PickleDB()
    with open(pkl_file, 'rb') as fp:
        soc_db_all = pickle.load(fp)

    train_dataset = SOCDataset(soc_db_all["train"],batch_len)
    test_dataset = SOCDataset(soc_db_all["test"],batch_len)

#    print(len(dataset))
#    print(dataset[100])
#    print(dataset[122:361])
    return train_dataset,test_dataset


# trd, tsd = GetSOCdata(batch_len=1, pkl=False)
# GetSOCdata(batch_len=1)
# for i, (inputs, soc_gt) in enumerate(trd):
#    print(i, (type(inputs),inputs, soc_gt))
