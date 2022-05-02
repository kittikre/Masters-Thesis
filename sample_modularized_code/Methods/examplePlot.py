# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:06:02 2021
Plot some examples

@author: Balint
"""

import numpy as np
from biosppy import timing

from Methods.dataHandling import getTargets, PTBXL_Beat_DataSet, PTBXL_DataSet
from Methods.plotECG import plotECG

np.random.seed(8)

def plotExamples(myPath, myLeads=['I'], classList=['MI'], nr_example=5, beat=True):
    timing.tic(name='example_plot_clk')
    myTargets = getTargets(path = myPath, classes = classList, subclass=False)
    # classDictPlot = classList
    if  beat:
        ptb_dataset= PTBXL_Beat_DataSet(path= myPath,leads=myLeads, target_list=myTargets[0])
    else:
        ptb_dataset= PTBXL_DataSet(path= myPath,leads=myLeads, target_list=myTargets[0])
    
    for i in range(len(ptb_dataset)):  
        x = int(np.random.choice(range(len(ptb_dataset))))
        print(x)
        sample = ptb_dataset[x]
        signal,label = sample['X'], sample['label']
    
        print(i, signal.shape, label)
        plotECG(signal.T,scale = 5,
                # title = 'Sample #{} from class '.format(i) + str(classDictPlot[int(label)]),
                norm ='nice')
        # plotECG(signal ,scale = 2.5, title = 'R-peaks owerlayed',rpeaks=[pre])
        if i == nr_example-1:
            # plt.show()
            break
    t1 = timing.tac(name='example_plot_clk')
    print('Plotting examples took %2i min %2i sec.'%(t1/60,t1%60))