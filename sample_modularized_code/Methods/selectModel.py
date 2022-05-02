# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 07:45:46 2021

For code readability a network selecter method
is implemented in this separate file

modelDict = {0: 'ConvNet', 1: 'Baloglu', 2:'LuiCNN',
             3: 'DeepCNN', 4: 'OgreCNN', 5: 'DenseNet', 
             6: 'ResNet18',7: 'ResNet152'}

@author: Balint
"""
import json


from Networks.Lui2018 import LuiCNN
from Networks.Baloglu2019 import BalogluNet
from Networks.Ogrezenau2020 import OgreCNN

from Networks.myCNN import ConvNet
from Networks.deepCNN import DeepCNN
from Networks.denseNet import DenseNet
from Networks.ResNet import resnet18, resnet152

def selectModel(netSelect, myPath, nr_leads, nr_classes, ecg_len):
    ''' 
    Method implementing the instance deployment of the selected model with 
    given network and training parameters
    
    Inputs:
        netSelect - the code for which model is selected (see modelDict)
        myPath
        nr_leads - the number of leads used
        nr_classes - the number of output classes
        ecg_len - the length of the ecg in seconds
    Outputs:
        model, network_params, training_params
    '''
    if netSelect==0:
        with open(myPath+'json/network_params.json', 'r') as jfile:
            network_params = json.load(jfile)
        with open(myPath+'json/training_params.json', 'r') as jfile:
            training_params = json.load(jfile)
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len
        model = ConvNet(network_params)
        
    elif netSelect==1:
        with open(myPath+'json/network_params_baloglu.json', 'r') as jfile:
            network_params = json.load(jfile)
        with open(myPath+'json/training_params_baloglu.json', 'r') as jfile:
            training_params = json.load(jfile)
        network_params['ch_conv'][0] = nr_leads
        network_params['lin_units'][-1] = nr_classes
        network_params['ecg_len'] = ecg_len
        model = BalogluNet(network_params)
        
    elif netSelect==2:
        network_params = {}
        network_params['hardcoded'] = 'yes'
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len
        with open(myPath+'json/training_params_Lui.json', 'r') as jfile:
            training_params = json.load(jfile)
        model = LuiCNN(ecg_len, nr_classes)
        
    elif netSelect==3:
        with open(myPath+'json/network_params_DConv.json', 'r') as jfile:
            network_params = json.load(jfile)
        with open(myPath+'json/training_params_DConv.json', 'r') as jfile:
            training_params = json.load(jfile)
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len
        model = DeepCNN(network_params)
    
    elif netSelect==4:
        network_params = {}
        network_params['hardcoded'] = 'yes'
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len
        with open(myPath+'json/training_params_Ogre.json', 'r') as jfile:
            training_params = json.load(jfile)
        model = OgreCNN(ecg_len, nr_classes)
    
    elif netSelect==5:
        with open(myPath+'json/network_params_denseNet.json', 'r') as jfile:
            network_params = json.load(jfile)
        with open(myPath+'json/training_params_denseNet.json', 'r') as jfile:
            training_params = json.load(jfile)
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len
        model = DenseNet(network_params)
        
    elif netSelect==6:
        network_params = {}
        network_params['hardcoded'] = 'yes'
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len       
        with open(myPath+'json/training_params_ResNet.json', 'r') as jfile:
            training_params = json.load(jfile)
        model = resnet18(nr_leads, nr_classes)
        
    elif netSelect==7:
        network_params = {}
        network_params['hardcoded'] = 'yes'
        network_params['in_ch'] = nr_leads
        network_params['n_class'] = nr_classes
        network_params['ecg_len'] = ecg_len       
        with open(myPath+'json/training_params_ResNet.json', 'r') as jfile:
            training_params = json.load(jfile)
        model = resnet152(nr_leads, nr_classes)
        
    return model, network_params, training_params