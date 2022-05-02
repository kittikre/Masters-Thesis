# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:33:58 2021

Ogrezenau 2020 - CNN

@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class OgreCNN(nn.Module):
    def __init__(self,ecg_len, nr_class):
        super(OgreCNN, self).__init__()
        self.ecg_len = ecg_len*100
        self.in_ch = 1
        self.nr_class = nr_class
        
        # self.l1_in_features = int(64*np.floor(self.ecg_len/2**3))
        self.l1_in_features = 64*int(self.ecg_len/2**3)                        # Since this number must alway be positive, truncation works as floor from numpy would
        
        self.convNet = nn.Sequential(
            nn.Conv1d(self.in_ch, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(64, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(64, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2))
        self.lin_class = nn.Sequential(
            nn.Linear(self.l1_in_features, self.nr_class),
            nn.Sigmoid())
        
    def forward(self, x):
        x = self.convNet(x)
        x = x.view(-1, self.l1_in_features)
        out = self.lin_class(x)
        return out