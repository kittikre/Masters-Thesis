# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:39:02 2021

class for deep convolutional neural network

@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self,network_params):
        super(DeepCNN, self).__init__()  
        ## Save parameters as own
        self.in_ch = network_params['in_ch']
        self.ecg_len = network_params['ecg_len']*100
        self.n_class = network_params['n_class']
        
        self.ch_conv = network_params['ch_conv']
        self.kernel_conv = network_params['kernel_conv']
        self.padding_conv = network_params['padding_conv']
        self.lin_units = network_params['lin_units']
        self.drop = network_params['dropout_level']
        
        self.pool_params = network_params['pool_params']                       # pool_params = [padding, dilation, kernel_size, stride]
        
        ## Preprocessing
        self.instNorm = nn.InstanceNorm1d(self.ecg_len)                        # Without Learnable Parameters
        
        ## Network: feature learning
        self.conv_block1 = self.conv_block_DConv(self.in_ch, self.ch_conv[0], kernel_size=self.kernel_conv[0],padding=self.padding_conv[0])
        self.conv_block2 = self.conv_block_DConv(self.ch_conv[0], self.ch_conv[1], kernel_size=self.kernel_conv[1],padding=self.padding_conv[1])
        self.conv_block3 = self.conv_block_DConv(self.ch_conv[1], self.ch_conv[2], kernel_size=self.kernel_conv[2],padding=self.padding_conv[2])
        self.conv_block4 = self.conv_block_DConv(self.ch_conv[2], self.ch_conv[3], kernel_size=self.kernel_conv[3],padding=self.padding_conv[3])
        self.conv_block5 = self.conv_block_DConv(self.ch_conv[3], self.ch_conv[4], kernel_size=self.kernel_conv[4],padding=self.padding_conv[4])
        
        ## Network: classification from features        
        # self.l1_in_features = int(self.ch_conv[4]*np.floor(self.ecg_len/2**len(self.ch_conv)))  # [l1_in_features] = [channels] * [length]
        self.l1_in_features = self.ch_conv[4]*int(self.ecg_len/2**len(self.ch_conv))  # Since value must be positive int() == np.floor() (int truncates)
        # [l1_in_features] = [channels] * [length]
        
        self.lin_class = nn.Sequential(
            nn.Linear(self.l1_in_features, self.lin_units[0]),
            # nn.BatchNorm1d(self.lin_units[0]),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.lin_units[0], self.lin_units[1]),
            # nn.BatchNorm1d(self.lin_units[1]),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.lin_units[1], self.lin_units[2]),
            # nn.BatchNorm1d(self.lin_units[2]),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.lin_units[2], self.n_class)
            )
        
    def forward(self, x):                                                      # x.size() = [batch, channel, length]
        x = self.instNorm(x)    
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        x = x.view(-1, self.l1_in_features)
        out = self.lin_class(x)
        return out
    
    def conv_block_DConv(self, in_f, out_f, *args, **kwargs):
        #=====================
        # Conv Block          
        #=====================
        # Convolutional 1D     
        # Batch normalisation 
        # Convolutional 1D     
        # Batch normalisation  
        # Max-pooling 1D        
        # Dropout 
        #=====================
        return nn.Sequential(
            nn.Conv1d(in_f, out_f,*args, **kwargs),
            # nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.Conv1d(out_f, out_f, *args, **kwargs),
            # nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.pool_params[2],stride=self.pool_params[3]),
            nn.Dropout(self.drop)
        )