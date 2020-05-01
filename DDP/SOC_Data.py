# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:17:40 2019

@author: H242998
"""

import scipy.io
import numpy as np
import pickle
#from torch.utils.data import Dataset
import os

pkl_file = './DATA/soc_db_v2.pkl'
###############################################################################

###############################################################################

class SOCLoader():
    def __init__(self,):
        self.data_folder = './DATA/Data_OLD/'
        self.pkl_file = pkl_file
        self.raw_files = {
                'train' : {
                        'cycle1' : '03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat',
                        'cycle2' : '03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat',
                        'cycle3' : '03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat',
                        'cycle4' : '03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat',
                        'udds' : '03-21-17_00.29 25degC_UDDS_Pan18650PF.mat',
                        'la92' : '03-21-17_09.38 25degC_LA92_Pan18650PF.mat',
                        'nn' : '03-21-17_16.27 25degC_NN_Pan18650PF.mat'
                        },
                
                'test' : {
                        'us06' : '03-20-17_01.43 25degC_US06_Pan18650PF.mat',
                        'hwfta' : '03-20-17_05.56 25degC_HWFTa_Pan18650PF.mat',
                        'hwftb' : '03-20-17_19.27 25degC_HWFTb_Pan18650PF.mat'
                        },
        #        'valid': {
        #                'cycle3' : '03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat',
        #                'la92' : '03-21-17_09.38 25degC_LA92_Pan18650PF.mat'
        #                }
                }
        
    def process(self,mat_file):
        data={}
        mat = scipy.io.loadmat(mat_file)['meas'][0][0]
        data['v'] = mat[1]
        data['i'] = mat[2]
        ah = mat[3]
        data['temp'] = mat[6]
        # tstamp = mat[0]
        # time = mat[7]
        # ctemp = mat[8]
        # wh = mat[4]
        # power = mat[5]
        data['soc'] = (4.2+ah)/4.2
        av = []
        ai = []
        average_factor = 400
        vlen = len(data['v'])
        
        for k in range(vlen+1):    
            if k-average_factor>=0:
                ai.append((sum(data['i'][k-average_factor:k]))/average_factor)
                av.append((sum(data['v'][k-average_factor:k]))/average_factor)
            elif k>0:
                ai.append((sum(data['i'][:k]))/k)
                av.append((sum(data['v'][:k]))/k)
        
        data['ai'] = np.array(ai)
        data['av'] = np.array(av)
        return data

    def CreateDB(self,):
        soc_db = {}
        for key in self.raw_files.keys():
            soc_db[key] = {}
            for dataset in self.raw_files[key].keys():
                mat_file = '{}{}'.format(self.data_folder,self.raw_files[key][dataset])
                soc_db[key][dataset] = self.process(mat_file)
            print("Processing complete from dataset: {}".format(dataset))
                        
        with open(self.pkl_file, 'wb+') as fp:
            pickle.dump(soc_db,fp,protocol=pickle.HIGHEST_PROTOCOL)
        

###############################################################################
def PickleDB():
    data_processor = SOCLoader()
    data_processor.CreateDB()

#class SOCDataset(Dataset):
class SOCDataset():
    def __init__(self, soc_db,batch_len):
        self.soc_db = soc_db
        self.samples = []
        self._init_dataset()
        if batch_len > 1:
            self.length_check(batch_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def length_check(self,batch_len):
        redn = len(self.samples)%batch_len
        if redn >= 1:
            del self.samples[-redn:]
        
        
    def _init_dataset(self):
        for key in self.soc_db.keys():
            subset = self.soc_db[key]
            for i in range(len(subset["v"])):
                self.samples.append((np.array([subset["v"][i][0],subset["temp"][i][0],subset["av"][i][0],subset["ai"][i][0]]),subset["soc"][i][0]))
            print("Loading complete from dataset: {}".format(key))

def GetSOCdata(batch_len = 1000,pkl = False):    
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

trd,tsd = GetSOCdata(batch_len = 1, pkl=False)
#for i, (inputs, soc_gt) in enumerate(trd):
#    print(i, (type(inputs),inputs, soc_gt))