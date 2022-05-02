# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 07:35:24 2021

A simple Neural Net with Dense layers
@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self,network_params):
        super(DenseNet, self).__init__()
        # Read in parameters
        self.in_ch = network_params['in_ch']
        self.ecg_len = network_params['ecg_len']*100
        self.n_class = network_params['n_class']
        self.drop = network_params['dropout_level']
        # Calculate input size
        self.in_len = int(self.in_ch*self.ecg_len)

        self.inst_norm = nn.InstanceNorm1d(self.ecg_len)        
        # Define classifier
        self.lin_class = nn.Sequential(
            nn.Linear(self.in_len, int(self.in_len//2)),
            nn.BatchNorm1d(int(self.in_len//2)),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(int(self.in_len//2), int(self.in_len//4)),
            nn.BatchNorm1d(int(self.in_len//4)),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(int(self.in_len//4), 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(8, self.n_class),
            # nn.Softmax(dim=1)
            )
    def forward(self, x):
        # x = self.inst_norm(x)
        x = x.view(-1, self.in_len)
        out = self.lin_class(x)
        return out