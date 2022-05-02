# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:54:20 2021

Baloglu 2019 - CNN

@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class BalogluNet(nn.Module):
    def __init__(self,network_params):
        super(BalogluNet, self).__init__()
        
        self.ecg_len = network_params['ecg_len']*100
        
        self.ch_conv = network_params['ch_conv']
        self.kernel_conv = network_params['kernel_conv']
        self.stride_conv = network_params['stride_conv']
        self.padding_conv = network_params['padding_conv']
        self.dilation_conv = network_params['dilation_conv']
        self.lin_units = network_params['lin_units']
        self.pool_params = network_params['pool_params']                       # pool_params = [padding, dilation, kernel_size, stride]
        self.drop = network_params['dropout_level']

        # MaxPooling - reused multiple times     
        self.maxpool = nn.MaxPool1d(kernel_size = self.pool_params[2],
                                    stride = self.pool_params[3],
                                    padding = self.pool_params[0],
                                    dilation = self.pool_params[1])
        # 1st conv layer     
        self.conv_1 = nn.Conv1d(in_channels=self.ch_conv[0],
                            out_channels=self.ch_conv[1],
                            kernel_size=self.kernel_conv[0],
                            stride=self.stride_conv[0], 
                            padding=self.padding_conv[0],
                            dilation=self.dilation_conv[0])
        
        # 2nd conv layer
        self.conv_2 = nn.Conv1d(in_channels=self.ch_conv[1],
                             out_channels=self.ch_conv[2],
                             kernel_size=self.kernel_conv[1],
                             stride=self.stride_conv[1],
                             padding=self.padding_conv[1],
                             dilation=self.dilation_conv[1])
        # Pool 1

        # 3rd conv layer
        self.conv_3 = nn.Conv1d(in_channels=self.ch_conv[2],
                             out_channels=self.ch_conv[3],
                             kernel_size=self.kernel_conv[2],
                             stride=self.stride_conv[2],
                             padding=self.padding_conv[2],
                             dilation=self.dilation_conv[2])

        # 4th conv layer
        self.conv_4 = nn.Conv1d(in_channels=self.ch_conv[3],
                             out_channels=self.ch_conv[4],
                             kernel_size=self.kernel_conv[3],
                             stride=self.stride_conv[3],
                             padding=self.padding_conv[3],
                             dilation=self.dilation_conv[3])
         # Pool 2
         
        
        # Keep track of dimension changes
        # L_out = ((L_in + 2*padding − dilation*(kernel_size−1)−1)/stride) +1
        self.in_length = self.ecg_len
        
        self.conv1_out_len = ((self.in_length + 2*self.padding_conv[0]-self.dilation_conv[0]*(self.kernel_conv[0]-1)-1)//self.stride_conv[0])+1                   
        self.conv2_out_len = ((self.conv1_out_len + 2*self.padding_conv[1]-self.dilation_conv[1]*(self.kernel_conv[1]-1)-1)//self.stride_conv[1])+1
 
        self.pool1_out_len = ((self.conv2_out_len + 2*self.pool_params[0]-self.pool_params[1]*(self.pool_params[2]-1)-1)//self.pool_params[3])+1

        self.conv3_out_len = ((self.pool1_out_len + 2*self.padding_conv[2] - self.dilation_conv[2]*(self.kernel_conv[2]-1)-1)//self.stride_conv[2])+1
        self.conv4_out_len = ((self.conv3_out_len + 2*self.padding_conv[3] - self.dilation_conv[3]*(self.kernel_conv[3]-1)-1)//self.stride_conv[3])+1
        
        self.pool2_out_len = ((self.conv4_out_len + 2*self.pool_params[0]-self.pool_params[1]*(self.pool_params[2]-1)-1)//self.pool_params[3])+1

        self.l1_in_features = int(self.ch_conv[4] * self.pool2_out_len)
        # [l1_in_features] = [channels] * [length]
             
        self.l_1 = nn.Linear(in_features=self.l1_in_features,
                          out_features=self.lin_units[0], # out_features=self.lin_units[0],
                          bias=True)
        
        self.l_out = nn.Linear(in_features=self.lin_units[0], 
                            out_features=self.lin_units[1],
                            bias=True)
        self.softmax = nn.Softmax(dim=1)
        # Dropout
        self.dropout = nn.Dropout(p=self.drop)

    def forward(self, x): # x.size() = [batch, channel, length]  

        # print("\nX shape: " + str(x.shape))
        # print("\n initial length: " + str(length))
        
        ## Conv1 
        x = nn.functional.relu(self.conv_1(x))
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv1: " + str(self.conv1_out_len))

        ## Conv2
        x = nn.functional.relu(self.conv_2(x))
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv2: " + str(self.conv2_out_len))

        ## Pool1
        x = self.dropout(self.maxpool(x))
        # print("\nX shape: " + str(x.shape))
        # print("\n length after pool2: " + str(self.pool2_out_len))
        
        ## Conv3
        x = nn.functional.relu(self.conv_3(x))
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv3: " + str(self.conv3_out_len))
        
        ## Conv4
        x = nn.functional.relu(self.conv_4(x))
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv4: " + str(self.conv4_out_len))

        ## Pool2
        x = self.maxpool(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after pool3: " + str(self.pool3_out_len))

        ## Linear classifier
        x = x.view(-1, self.l1_in_features)     # Flatten the last conv layer

        x = self.dropout(nn.functional.relu(self.l_1(x)))
        output = self.softmax(self.l_out(x))
        # output = self.l_out(x)
        return output