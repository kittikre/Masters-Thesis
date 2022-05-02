# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:29:50 2021

Lui et al 2018 - CNN

@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class LuiCNN(nn.Module):
    def __init__(self,ecg_len,nr_classes):
        super(LuiCNN, self).__init__()
        self.ecg_len = ecg_len*100
        self.nr_classes = nr_classes
        #=====================================
        #layer  Name                out Shape
        #=====================================
        # 0     Inputs              512         1000
        # 1–6   Convolutional block 256 × 32    500 x 32
        # 7–12  Convolutional block 128 × 32    250 x 32
        # 13–18 Convolutional block 64 × 32     125 x 32
        # 19–24 Convolutional block 32 × 32     62 x 32     ? or 63
        # 25    Flattened           1024        1984        ? or 2016
        # 26    Fully connected     32          32
        # 27    Batch normalisation 32
        # 28    Dropout 50%         32
        # 29    Fully connected     32
        # 30    Batch normalisation 32
        # 31    Dropout 50%         32
        # 32    Fully connected     16
        # 33    Batch normalisation 16
        # 34    Dropout 50%         16
        # 35    Outputs             4
        #====================================
       
        self.instNorm = nn.InstanceNorm1d(self.ecg_len)  # Without Learnable Parameters       
        self.conv_block1 = self.conv_block_Lui(1, 32, kernel_size=3,padding=1)
        self.conv_block2 = self.conv_block_Lui(32, 32, kernel_size=3,padding=1)
        self.conv_block3 = self.conv_block_Lui(32, 32, kernel_size=3,padding=1)
        self.conv_block4 = self.conv_block_Lui(32, 32, kernel_size=3,padding=1)
        
        # [l1_in_features] = [channels] * [length]
        # self.l1_in_features = int(32*np.floor(self.ecg_len/2**4))
        self.l1_in_features = 32*int(self.ecg_len/2**4)                        # Since value must be positive int() == np.floor() (int truncates)
        
        self.lin_class = nn.Sequential(
            nn.Linear(self.l1_in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(16, 3)
            nn.Linear(16, self.nr_classes)
            )
        
    def forward(self, x):# x.size() = [batch, channel, length]
        x = self.instNorm(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = x.view(-1, self.l1_in_features)
        out = self.lin_class(x)
        return out
        
    def conv_block_Lui(self, in_f, out_f, *args, **kwargs):
        #====================================
        # Conv Block            ch    kernel
        #====================================
        # Convolutional 1D      32      3
        # Batch normalisation   –       –
        # Convolutional 1D      32      3
        # Batch normalisation   –       –
        # Max-pooling 1D        -       2
        # Dropout 50%
        #====================================
        return nn.Sequential(
            nn.Conv1d(in_f, out_f,*args, **kwargs),
            nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.Conv1d(out_f, out_f, *args, **kwargs),
            nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Dropout(p=0.5)
        )