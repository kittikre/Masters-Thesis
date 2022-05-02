# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 07:27:17 2021
This file contains methods to perform heartbeat segmentation on ECG signal
The methods defined in this file:   
    Method to find beat info: [R-peak, beat_start, beat_end]
    Method to extract beats, given beat info and signal
    Method to get R-peaks for a list of recordings
    Method to get a heartbeat given an R-peak

@author: Balint
"""

import numpy as np
import pandas as pd
import ast
import wfdb

from biosppy.signals import ecg
from Methods.plotECG import plotECG


np.random.seed(8)

#%% 
def findBeatInfo(signal, pre=0.3, freq=100): #[R-peak,beat_start,beat_end]
    pre = pre*freq
    try:
        _,_,rpeaks,_,_,_,_ = ecg.ecg(signal,freq,show=False)
    except:
        print('Recording was skipped due to an error')
        return 'Error'
    
    # Make a flat list of R-peaks start and end of beats
    rp_list = rpeaks.tolist()
    last_rp = len(rp_list)
    beat_info = []
    [beat_info.append([int(rp_list[rp_idx]),int(rp_list[rp_idx]-pre),int(rp_list[rp_idx+1]-pre)]) for rp_idx in range(0,last_rp-1) if int(rp_list[rp_idx]-pre)>0]
    beat_info.append([int(rp_list[-1]),int(rp_list[-1]-pre),int(1000)])
    return beat_info

def getBeat(signal, beat_info, ecg_len, nr_leads,freq=100,myPad='edge',
            allowOverlap=0):
    # Extract heartbeats
    # Define beat boundaries
    beat_len = int(ecg_len*freq)
    start = int(beat_info[1])
    if allowOverlap:
        end = int(beat_info[0]+(beat_len-(beat_info[0]-beat_info[1])))
    else:
        end = int(beat_info[2])
    
      
    # Preallocate X
    X = np.zeros([nr_leads,beat_len])

    # Truncate
    if (end-start) > beat_len:
        end = int(start+beat_len)
        for i in range(nr_leads):
            X[i] = signal[0, i, start:end]
    # Zero-pad -> pad with last value
    elif (end-start) < beat_len:
        nr_zeros = beat_len-(end-start)
        for i in range(nr_leads):
            X[i] = np.pad(signal[0, i, start:end],(0,nr_zeros),mode=myPad)
    # Do nothing, size is perfect
    else:
        for i in range(nr_leads):
            X[i] = signal[0, i, start:end]            
    return X
    
def rpeaksFromList(path, input_list, pre_s = 0.3, ecg_lead=10, freq=100,
                   plotIt=0, flatten=1):
    pre_s = pre_s*freq
    # Get Y_raw 
    Y_raw = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y_raw.scp_codes = Y_raw.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    rpeaks_list = []
    error_idx = []
    for idx, val in enumerate(input_list):
        print(idx)
        # Get X_raw
        X_raw = [wfdb.rdsamp(path+Y_raw['filename_lr'].iloc[val])]
        X_raw = np.array([signal for signal, meta in X_raw]) 
        # shape: [1,1000,12]
        signal = X_raw[0,:,ecg_lead]

        
        # Get R-peaks
        try:
            _,_,rpeaks,_,_,_,_ = ecg.ecg(signal,freq,show=False)
        except:
            error_idx += [val]
            print('Recording with ecg_id '+str(val)+
                  ' was skipped due to an error')
            continue
        
        # Visualize
        if plotIt:
            plotECG(signal ,scale = 2.5, title = 'R-peaks owerlayed',
                    rpeaks=rpeaks/100)
        
        # Save rpeaks into a list of lists
        if flatten:
            # transforms it to a flat list with [ecg_id, rpeak, beat_start, beat_end]
            rp_list = rpeaks.tolist()
            last_rp = len(rp_list)
            [rpeaks_list.append([input_list[idx],int(rp_list[rp_idx]),int(rp_list[rp_idx]-pre_s),int(rp_list[rp_idx+1]-pre_s)]) for rp_idx in range(0,last_rp-1) if int(rp_list[rp_idx]-pre_s)>0]
            rpeaks_list.append([input_list[idx],int(rp_list[-1]),
                                int(rp_list[-1]-pre_s),int(1000)])

        else:
            rpeaks_list += [rpeaks.tolist()]

    return rpeaks_list
    
def getBeatfrom_myInput(path, myInput,ecg_lead=10, freq=100,plotIt=0,
                        pre=0.2,post=0.4):
    
    ecg_id , rpeak = myInput
    start = int(rpeak - freq*pre)
    end = int(rpeak + freq*post)
    
    Y_raw = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y_raw.scp_codes = Y_raw.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    X_raw = [wfdb.rdsamp(path+Y_raw['filename_lr'].iloc[ecg_id])]
    X_raw = np.array([signal for signal, meta in X_raw]) # shape: [1,1000,12]
    
    signal = X_raw[0,start:end,ecg_lead]
    
    if plotIt:
        plotECG(signal ,scale = 2.5, title = 'R-peaks owerlayed',rpeaks=[pre])
        
    return signal