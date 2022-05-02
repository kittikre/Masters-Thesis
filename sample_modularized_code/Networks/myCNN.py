# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:08:58 2021

A convolutional neural network inspiried by the BalogluNet

@author: Balint
"""
import torch
torch.manual_seed(8)

import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self,network_params):
        super(ConvNet, self).__init__()
        ## Saving parameters
        self.in_ch = network_params['in_ch']
        self.ecg_len = network_params['ecg_len']*100
        self.n_class = network_params['n_class']
        
        self.ch_conv = network_params['ch_conv']
        self.kernel_conv = network_params['kernel_conv']
        self.stride_conv = network_params['stride_conv']
        self.padding_conv = network_params['padding_conv']
        self.dilation_conv = network_params['dilation_conv']
        self.lin_units = network_params['lin_units']
        self.pool_params = network_params['pool_params']                       # pool_params = [padding, dilation, kernel_size, stride]
        self.drop = network_params['dropout_level']
        
        ## Units of the Network
        
        # MaxPooling - reused multiple times
        self.maxpool = nn.MaxPool1d(kernel_size = self.pool_params[2],
                                    stride = self.pool_params[3],
                                    padding = self.pool_params[0],
                                    dilation = self.pool_params[1])
        # 1st conv layer
        self.conv_1 = nn.Conv1d(in_channels=self.in_ch,
                            out_channels=self.ch_conv[0],
                            kernel_size=self.kernel_conv[0],
                            stride=self.stride_conv[0], 
                            padding=self.padding_conv[0],
                            dilation=self.dilation_conv[0])   

        # Batch Norm 1
        self.bn1 = nn.BatchNorm1d(self.ch_conv[0])

        # Pool 1
        
        # 2nd conv layer
        self.conv_2 = nn.Conv1d(in_channels=self.ch_conv[0],
                             out_channels=self.ch_conv[1],
                             kernel_size=self.kernel_conv[1],
                             stride=self.stride_conv[1],
                             padding=self.padding_conv[1],
                             dilation=self.dilation_conv[1])
        # Batch Norm 2
        self.bn2 = nn.BatchNorm1d(self.ch_conv[1])

        # Pool 2

        # 3rd conv layer
        self.conv_3 = nn.Conv1d(in_channels=self.ch_conv[1],
                             out_channels=self.ch_conv[2],
                             kernel_size=self.kernel_conv[2],
                             stride=self.stride_conv[2],
                             padding=self.padding_conv[2],
                             dilation=self.dilation_conv[2])
        # Batch Norm 3
        self.bn3 = nn.BatchNorm1d(self.ch_conv[2])

        # Pool 3

        # 4th conv layer
        self.conv_4 = nn.Conv1d(in_channels=self.ch_conv[2],
                             out_channels=self.ch_conv[3],
                             kernel_size=self.kernel_conv[3],
                             stride=self.stride_conv[3],
                             padding=self.padding_conv[3],
                             dilation=self.dilation_conv[3])
        
        # Batch Norm 4
        self.bn4 = nn.BatchNorm1d(self.ch_conv[3])
        
        # Keep track of dimension changes
        # L_out = ((L_in + 2*padding − dilation*(kernel_size−1)−1)/stride) +1
        self.conv1_out_len = ((self.ecg_len + 2*self.padding_conv[0]-self.dilation_conv[0]*(self.kernel_conv[0]-1)-1)//self.stride_conv[0])+1
        self.pool1_out_len = ((self.conv1_out_len + 2*self.pool_params[0]-self.pool_params[1]*(self.pool_params[2]-1)-1)//self.pool_params[3])+1
                   
        self.conv2_out_len = ((self.pool1_out_len + 2*self.padding_conv[1] - self.dilation_conv[1]*(self.kernel_conv[1]-1)-1)//self.stride_conv[1])+1
        self.pool2_out_len = ((self.conv2_out_len + 2*self.pool_params[0]-self.pool_params[1]*(self.pool_params[2]-1)-1)//self.pool_params[3])+1

        self.conv3_out_len = ((self.pool2_out_len + 2*self.padding_conv[2] - self.dilation_conv[2]*(self.kernel_conv[2]-1)-1)//self.stride_conv[2])+1
        self.pool3_out_len = ((self.conv3_out_len + 2*self.pool_params[0]-self.pool_params[1]*(self.pool_params[2]-1)-1)//self.pool_params[3])+1

        self.conv4_out_len = ((self.pool3_out_len + 2*self.padding_conv[3] - self.dilation_conv[3]*(self.kernel_conv[3]-1)-1)//self.stride_conv[3])+1    

        self.l1_in_features = int(self.ch_conv[3] * self.conv4_out_len)
        # [l1_in_features] = [channels] * [length]
        
        self.l_1 = nn.Linear(in_features=self.l1_in_features,
                          out_features=self.lin_units[0], # out_features=self.lin_units[0],
                          bias=True)
        
        self.l_2 = nn.Linear(in_features=self.lin_units[0],
                          out_features=self.lin_units[1],
                          bias=True)
        
        self.l_3 = nn.Linear(in_features=self.lin_units[1],
                          out_features=self.lin_units[2],
                          bias=True)
        
        self.l_out = nn.Linear(in_features=self.lin_units[2], 
                            out_features=self.n_class,
                            bias=True)
        
        # add dropout to network
        self.dropout = nn.Dropout(p= self.drop)
    
    def forward(self, x): # x.size() = [batch, channel, length]  

        # print("\nX shape: " + str(x.shape))
        # print("\n initial length: " + str(length))
        
        ## Conv1        
        x = self.conv_1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv1: " + str(self.conv1_out_len))

        ## Pool1
        x = self.maxpool(x)
        # x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after pool1: " + str(self.pool1_out_len))

        ## Conv2        
        x = self.conv_2(x)
        x = self.bn2(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv2: " + str(self.conv2_out_len))

        ## Pool2
        x = self.maxpool(x)
        # x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after pool2: " + str(self.pool2_out_len))
        
        ## Conv3 
        x = self.conv_3(x)
        x = self.bn3(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv3: " + str(self.conv3_out_len))

        ## Pool3
        x = self.maxpool(x)
        # x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after pool3: " + str(self.pool3_out_len))
        
        ## Conv4        
        x = self.conv_4(x)
        x = self.bn4(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        # print("\nX shape: " + str(x.shape))
        # print("\n length after conv4: " + str(self.conv4_out_len))

        ## Linear classifier      
        x = x.view(-1, self.l1_in_features)     # Flatten the last conv layer
        # print(x.shape)       
        x = nn.ReLU(self.l_1(x))
        x = self.dropout(x)
        x = nn.ReLU(self.l_2(x))
        x = nn.ReLU(self.l_3(x))
        output = self.l_out(x)
        #output = softmax(self.l_out(x), dim=1)
        
        return output