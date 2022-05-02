# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 08:36:39 2021
test loop multi model for ROC

@author: Vespa
"""
import os
# import datetime
import random
import numpy as np
from biosppy import timing

import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
# from copy import deepcopy

from Methods.dataHandling import getTargets, PTBXL_DataSet  # ,PTBXL_Beat_DataSet
from Methods.selectModel import selectModel
from Methods.resultEval import confMatrix, acc_measures, roc_auc, count_parameters, compare_roc # learningCurve
# from Methods.beatSegmentation import findBeatInfo, getBeat

# myPath =  'C:\\Users\\admin\\Desktop\\Balint\\' 
myPath = os.getcwd()+'\\'

# Reproducability parameters
torch.manual_seed(8)
np.random.seed(8)
g = torch.Generator()
g.manual_seed(8)

rocMultiList = []
legendList = []

#%% 1 - Create dataSets
### Baloglu
# excludeList=[]
# classList = ['AMI','IMI','LMI','PMI','NORM']; sClass = True
# classList = ['AMI','IMI','NORM']; sClass = True
# classList = ['MI','NORM']; sClass = False

# classList = ['MI', 'NORM']; sClass = False
# excludeList=  [['MI'],['LMI','PMI']]# [[category to exclude from],[subcategories to exclude]]

# ### Ogrezeanu
# excludeList = [['STTC'],['NST_','STTC',]]
# classList = ['STTC','NORM']; sClass = False

### Lui
# excludeList=[]
# classList = ['MI','NORM','Other']; sClass = False

### DeepCNN
excludeList = []
# classList = ['MI','NORM']; sClass = False
classList = ['MI','Other']; sClass = False
# classList = ['MI','NORM','STTC','CD','HYP']; sClass = False

if len(excludeList) == 0:
    _, _, testList = getTargets(path = myPath, classes=classList,
                                            subclass=sClass)
else:
    _, _, testList = getTargets(path = myPath, classes=classList,
                                            subclass=sClass,exclude_list=excludeList) 

nr_classes = len(classList)
print('target lists created')
#%% 2 - Import model
modelDict = {0: 'ConvNet', 1: 'Baloglu', 2:'LuiCNN',
             3: 'DeepCNN', 4: 'OgreCNN', 5: 'DenseNet', 
             6: 'ResNet18',7: 'ResNet152'}

# ecg_leads = ['II','III','aVF']      # Reasat
# ecg_leads = ['V5']                  # WARD


## Baloglu
# netSelect = 1
# ecg_leads = ['V4']
# model_id = 'DenseNet_2021-07-09_07h22'

# ## Lui et al
# netSelect = 2
# ecg_leads = ['I']

         
## Ogre
# netSelect = 4
# ecg_leads = ['V4']


## DenseNet
# netSelect = 5
# ecg_leads = ['V5']


## DeepCNN
netSelect = 3
ecg_leads = ['V5']
# model_id = 'DeepCNN_2021-07-09_12h20' #DeepCNN, MIvsNORM, no segment, V5 best result
model_id = 'DeepCNN_2021-07-19_10h20' #DeepCNN, MIvsOther, no segment, all leads best result 
 
# ecg_leads = ['I','II','III']
# model_id = 'DeepCNN_2021-07-13_07h55' #DeepCNN, MIvsNORM, no segment, Roman leads best result 
# ecg_leads = ['all']
# model_id = 'DeepCNN_2021-07-11_17h33' #DeepCNN, MIvsNORM, no segment, all leads best result 

# Override nr of classes
nr_classes = 2


## ResNet
# netSelect = 6
# ecg_leads = ['V5']
# gracePeriod = 10

# Calculate length of ecg
ecg_len = 10    #[sec]

# Calculate number of leads
if ecg_leads == ['all']:
    nr_leads = 12
else:
    nr_leads = len(ecg_leads)
    
model, network_params, training_params = selectModel(netSelect, myPath, nr_leads,
                                                     nr_classes, ecg_len)

# Count the number of trainable parameters in the model
table,total_params = count_parameters(model)
# # How many GPUs are there?
# print(torch.cuda.device_count())

cuda = torch.cuda.is_available() 
device = torch.device("cuda:0" if cuda else "cpu")

model_path = myPath+'Results\\' + model_id + '\\' 

model,_,_ = selectModel(netSelect, myPath, nr_leads, nr_classes, ecg_len)
model.load_state_dict(torch.load(model_path+model_id+'.pth',map_location=device))
model.to(device)
model.eval()
#%% 4 - Cuda helper functions
def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
#%% 8 - Testing
timing.tic(name='test')

## Load best model
model,_,_ = selectModel(netSelect, myPath, nr_leads, nr_classes, ecg_len)
model.load_state_dict(torch.load(model_path+model_id+'.pth',map_location=torch.device(device)))
model.to(device)
model.eval()

## Make testing dataset and dataloader
test_ptb_dataset= PTBXL_DataSet(path= myPath, target_list=testList,
                                leads=ecg_leads, upSample='No')
test_dataloader = DataLoader(test_ptb_dataset,shuffle=False,
                             worker_init_fn=seed_worker, generator=g) # batch_size=1,

# Init for full segment
true_seg = []
pred_seg = []
pred_seg_prob = []

softMax = Softmax(dim=1)
for i, data in enumerate(test_dataloader, 0):
    sample = data
    test_features, test_labels = sample['X'], sample['label']
    X_batch = get_variable(test_features).float()                           # Converts tensor to cuda
        
    output = model(X_batch)
    preds = torch.max(output, 1)[1]
    output_prob = softMax(output)
        
    true_seg += list(test_labels)
    pred_seg += list(get_numpy(preds.data))                               # Converts tensor to Numpy   
    pred_seg_prob += [get_numpy(output_prob)[0,-1]]   
    
test_acc = balanced_accuracy_score(true_seg, pred_seg)
t_test = timing.tac(name='test')
print('Testing took %2i min %2i sec'%(t_test/60,t_test%60))
#%% 9 - Results
## Confusion Matrix, Accuracy measures
if len(classList)==2:
    roc_seg = roc_auc(true_seg, pred_seg_prob,plotIt=0)
    rocList = [roc_seg]
    mylegend=['Segment AUC: %0.2f' %(roc_seg[-1])]  
    # compare_roc(rocList,mylegend,figPath=model_path+'test_roc.png')

rocMultiList +=rocList
legendList += mylegend
compare_roc(rocMultiList,legendList)

# sb = 0 # parameter to split the barplot according to classes

# confM =confMatrix(true_seg, pred_seg ,classList,nr_classes=nr_classes,figPath=model_path+'test_confM.png')
# results =  acc_measures(confM,true_seg, pred_seg ,
#                         figPath=model_path+'test_results.png',plotIt=1,splitBars=sb)

# table,total_params = count_parameters(model)

# #%% 10 - Save logs
# # comment = 'With softmax'
# with open(model_path+'info_test.txt', 'w') as f:
#     f.write('Model: '+ modelDict[netSelect])
#     f.write('\nLeads used: '+str(ecg_leads))
#     f.write('\nClasses: '+str(classList))
#     f.write('\n Exclude list: '+str(excludeList))
#     f.write('\nTrainable params:'+str(total_params))
#     f.write('\n')
#     f.write('\nTraining hyperparameters\n')
#     for key, value in training_params.items():  
#         f.write('- %s: %s\n' % (key, value))
    
#     f.write('\nNetwork parameters\n')
#     for key, value in network_params.items():  
#         f.write('- %s: %s\n' % (key, value))
        
#     f.write('\n')
#     f.write('Training and validation did not run')
#     f.write('\nResults: \n'+str(results))
#     # f.write(comment)
# print('Script is finished, logs are saved')