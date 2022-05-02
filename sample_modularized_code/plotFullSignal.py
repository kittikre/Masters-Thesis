# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:39:32 2021

@author: Vespa
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator

from Methods.dataHandling import getTargets, PTBXL_DataSet

myLeads=['I']
classList=['NORM']
nr_example=50

myPath = os.getcwd()+'\\'
sFreq = 100
ratio = 0.03937007874015748             # To convert mm to inch 

myTargets = getTargets(path = myPath, classes = classList, subclass=False)
ptb_dataset= PTBXL_DataSet(path= myPath,leads=myLeads, target_list=myTargets[0])

#%%
for i in range(len(ptb_dataset)):  
    x = int(np.random.choice(range(len(ptb_dataset))))
    print(x)
    sample = ptb_dataset[x]
    signal,label = sample['X'], sample['label']

    print(i, signal.shape, label)
    
    ECG = signal.T
    ECG_len = len(ECG)/sFreq                # The length of the signal in seconds
    
    t = np.arange(0,ECG_len,0.01)
    yrange = float((abs(min(ECG)*1.1) +abs(max(ECG))*1.1)*10)
    
    
    fig = plt.figure(figsize=(400*ratio,150*ratio))
    ax = fig.add_subplot()
    ax.plot(t,ECG,linewidth=2)
    plt.ylim([min(ECG)*1.1, max(ECG)*1.1])
    plt.ylabel("Amplitude [mV]")
    plt.xlim([0 ,ECG_len])
    plt.xlabel("Time [s]") 
    
    
    plt.rc('font', size=15) 
    plt.minorticks_on()
    
    plt.show()
    if i == nr_example-1:
        # plt.show()
        break