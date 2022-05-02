# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:38:06 2021
Plot single lead ECG recording
    The plot is fixed in height but has an adaptive length
    x axis =  time: 1mm is 0.04 sec (thin), 5mm is 0.2 sec (thick)
    y axis = milliVolts: 1 mm is 0.1 mV (thin), 5mm is 0.5 mV (thick)
Inputs:
    signal: the values in an ECG (array of float64)
    sFreq: sampling frequency (int) , default is 100 Hz
    scale: scaling of the plot (int)
    yAx: limits of the y-axis
    fontScale
    norm: if it is True, the signal is normalized,if not,it is plotted with
        fixed values on the y axis, if something else, the plot is adjusted
        to the spread of the signal
    title: takes the title of the plot
    rpeaks: Given a list it plots vertical red lines at the locations
Outputs:
    A figure, returns nothing

@author: Vespa
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator # AutoMinorLocator

np.random.seed(8)

def plotECG(signal,sFreq=100,scale= 5,yAx=[-1.2,1.8],fontScale=1,
            norm=False,title=False, rpeaks=0):
    ECG = signal
    ECG_norm = ECG/max(ECG)
    
    ECG_len = len(ECG)/sFreq           # The length of the signal in seconds
    t = np.arange(0,ECG_len,0.01)
    

    ratio = 0.03937007874015748             # To convert mm to inch 

    if norm == True:
        yrange = float((abs(min(ECG_norm)*1.1)+1.1)*10)
        fig = plt.figure(figsize=(np.ceil(ECG_len/0.04*ratio*scale),
                                  np.ceil(yrange*ratio*scale)))
        ax = fig.add_subplot()
        ax.plot(t,ECG_norm)
        plt.ylim([min(ECG_norm)*1.1, 1.1])
        plt.ylabel("Amplitude [-]")
        plt.xlim([0 ,ECG_len])
        plt.xlabel("Time [s]")
    elif norm == False:
        yrange = float(sum([abs(elem) for elem in yAx])*10)
        fig = plt.figure(figsize=(np.ceil(ECG_len/0.04*ratio*scale),
                                  np.ceil(yrange*ratio*scale))) 
        ax = fig.add_subplot()
        ax.plot(t,ECG)
        plt.ylim(yAx)
        plt.ylabel("Amplitude [mV]")
        plt.xlim([0 ,ECG_len])
        plt.xlabel("Time [s]")  
    else:
        yrange = float((abs(min(ECG)*1.1) +abs(max(ECG))*1.1)*10)
        fig = plt.figure(figsize=(np.ceil(ECG_len/0.04*ratio*scale),
                                  np.ceil(yrange*ratio*scale))) 
        ax = fig.add_subplot()
        ax.plot(t,ECG)
        plt.ylim([min(ECG)*1.1, max(ECG)*1.1])
        plt.ylabel("Amplitude [mV]")
        plt.xlim([0 ,ECG_len])
        plt.xlabel("Time [s]") 
        
    if title:
        plt.title(title)
        
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    plt.rc('font', size=np.ceil(12*fontScale)) 
    # ax.rc('lines',linewidth=3)
    plt.grid(which = 'major', color='red', linestyle='-.', linewidth=0.8)
    plt.grid(b = True, which = 'minor', color='red', linestyle='-.',
             linewidth=0.8,alpha=0.2)
    plt.minorticks_on()
    # if rpeaks != 0:
    plt.vlines(rpeaks,yAx[0],yAx[1],colors='red')
    plt.show()