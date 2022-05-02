# -*- coding: utf-8 -*-
"""Main script - hearbeat segmentation

This is the main script for MI detection in ECG with Deep Learning, using
heartbeat segmented data.
It is made up of ten blocks, these are:
    1 - Create Data sets
    2 - Import model
    3 - Define loss function and optimizer
    4 - Helper functions
    5 - Set up logging
    6 - Training and Validation
    7 - Testing
    8 - Results
    9 - Print learning curve
    10 - Save logs

To run it requires to have all the imported methods and their dependencies
from the directory 'Methods', and the PTB-XL 100Hz data from the 'records100' 
folder. 

Last updated on July 21 2021
@author: Vespa
"""
import os
import datetime
import random
import numpy as np
from biosppy import timing

import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from copy import deepcopy

from Methods.dataHandling import getTargets, PTBXL_Beat_DataSet, PTBXL_DataSet
from Methods.selectModel import selectModel
from Methods.resultEval import learningCurve, confMatrix, acc_measures, roc_auc, count_parameters, compare_roc
from Methods.beatSegmentation import findBeatInfo, getBeat

myPath = os.getcwd()+'\\'


# Reproducability parameters
torch.manual_seed(8)
np.random.seed(8)

# #%% Plot some examples
# from Methods.examplePlot import plotExamples
# plotExamples(myPath,['V5'], nr_example=3)

#%% 1 - Create dataSets
### Baloglu
# excludeList = []
# classList = ['AMI','IMI','LMI','PMI','NORM']; sClass = True
# classList = ['AMI','IMI','NORM']; sClass = True
# classList = ['MI','NORM']; sClass = False

# classList = ['MI', 'NORM']; other = 0 ; sClass = False
# excludeList=  [['MI'],['LMI','PMI']]# [[category to exclude from],[subcategories to exclude]]

### Ogrezeanu
# excludeList = [['STTC'],['NST_','STTC',]]
# classList = ['STTC','NORM']; sClass = False

### Lui
excludeList = []
# classList = ['MI','NORM','Other']; sClass = False
classList = ['MI','NORM']; sClass = False

### DenseNet and DeepCNN
# excludeList = []
# excludeList=  [['MI'],['LMI','PMI']]
# classList = ['MI', 'NORM']; other = 0 ; sClass = False

if len(excludeList) == 0:
    trainList, validList, testList = getTargets(path = myPath, classes=classList,
                                            subclass=sClass)
else:
    trainList, validList, testList = getTargets(path = myPath, classes=classList,
                                            subclass=sClass,exclude_list=excludeList)    

nr_classes = len(classList)
print('target lists created')

#%% 2 - Import model
modelDict = {0: 'ConvNet', 1: 'Baloglu', 2:'LuiCNN',
             3: 'DeepCNN', 4: 'OgreCNN', 5: 'DenseNet', 
             6: 'ResNet18',7: 'ResNet152'}

# ecg_leads = ['II','III','aVF']      # Reasat
# ecg_leads = ['V5']                  # WARD
# ecg_leads = ['all']
# ecg_leads = ['I','II','III']

# ## Baloglu - should turn off the nice segmentation
# netSelect = 1
# ecg_leads = ['V4']
# pre_R = 0.25
# post_R = 0.4                
# padding = 'edge' # last value
# overlap = 1

## Lui et al
# netSelect = 2
# ecg_leads = ['I']
# pre_R = 0
# post_R = 1.7 
# padding = 'constant'    # zeros
# overlap = 0
# gracePeriod = 50
         
## OgreCNN
# netSelect = 4
# ecg_leads = ['V4']
# pre_R = 0.3               # 0.3 [sec]
# post_R = 0.5              # 0.5 [sec]
# padding = 'edge'          # last value
# overlap = 0
# gracePeriod = 100

## DenseNet
# netSelect = 5
# ecg_leads = ['V5']
# pre_R = 0.3               # 0.3 [sec]
# post_R = 0.5              # 0.5 [sec]
# padding = 'edge'          # last value
# overlap = 0
# gracePeriod = 50

## DeepCNN
# # netSelect = 3
ecg_leads = ['V5']
pre_R = 0.3               # 0.3 [sec]
post_R = 0.5              # 0.5 [sec]
padding = 'edge'          # last value
overlap = 0
# gracePeriod = 20


## ResNet
netSelect = 7
# ecg_leads = ['V5']
gracePeriod = 15

# Calculate length of ecg
ecg_len = pre_R + post_R    #[sec]

# Calculate number of leads
if ecg_leads == ['all']:
    nr_leads = 12
else:
    nr_leads = len(ecg_leads)
    
model, network_params, training_params = selectModel(netSelect, myPath,
                                                     nr_leads, nr_classes,
                                                     ecg_len)

# Count the number of trainable parameters in the model
table,total_params = count_parameters(model)
# # How many GPUs are there?
# print(torch.cuda.device_count())

cuda = torch.cuda.is_available() 
device = torch.device("cuda:0" if cuda else "cpu")
# print("Using device:", device)

if cuda:
    model = model.to(device)
    print('Model is imported and transfered to Cuda')
print(model)

#%% 3 - Define loss function and optimizer
import torch.optim as optim
from torch.nn import CrossEntropyLoss

learning_rate = training_params['learning_rate']

criterion = CrossEntropyLoss()

if netSelect in [2,4]: # Ogrezenau, Lui
    optimizer = optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.9,
                              eps=1e-07)  # Ogre

elif netSelect in [0,1,3,5]: # Baloglu, DenseNet, DeepCNN
    w_decay = training_params['w_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=w_decay)

elif netSelect in [6,7]: # ResNet 
    mom = training_params['momentum']
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=mom)

# Reproducability parameters
g = torch.Generator()
g.manual_seed(8)
# torch.use_deterministic_algorithms()

# #%% Test the forward pass with dummy data
# # import torch
# from torch.autograd import Variable
# x = np.random.normal(0 , 1, (4, 1, 1000)).astype('float32')
# x = Variable(torch.from_numpy(x))
# if torch.cuda.is_available():
#     x = x.cuda()
# out = model(x)
# print(out.size())
# print(out)

#%% 4 - Helper functions
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

def printConfig():
    print('\nModel: '+ modelDict[netSelect])
    print('* Leads used: '+str(ecg_leads))
    print('* Classes: '+str(classList))
    print('* Exclude list: '+str(excludeList)) 
    print('* Trainable params:'+str(total_params))
    # Hyperparams
    print('\n Training hyperparameters')
    for key, value in training_params.items():  
        print('* %s: %s' % (key, value))
        
    # Network architecture  
    print('\n Network architecture')
    for key, value in network_params.items():  
        print('* %s: %s' % (key, value))

def resample(dataSet, upS):
    if upS:
        dataSet.upSampleMinor()
    else:
        dataSet.downSampleMajor()
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#%% 5 - Set up logging
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
# user = 'Balint-199'
model_id = f'{modelDict[netSelect]}_{current_time}'
model_path = myPath+'Results\\' + model_id + '\\'
try:
    os.makedirs(model_path, exist_ok = True)
except OSError as error:
    print(error)

#%% 6 - Training and Validation
# Training parameters
batch_size = training_params['batch_size']  
num_epochs = training_params['num_epochs']
upSample = training_params['upSample']
early_stop = training_params['early_stop']
early_stop_limit = training_params['early_stop_limit']

printConfig()

# Init variables
train_acc, train_loss = [], []
valid_acc = []
early_stop_counter = 0
best_model_state = False

timing.tic(name='dataSetCreate')

train_ptb_dataset= PTBXL_Beat_DataSet(path=myPath,target_list=trainList, 
                                      leads=ecg_leads, upSample=upSample,
                                      pre=pre_R, post=post_R,myPad=padding,
                                      allowOverlap=overlap)
train_dataloader = DataLoader(train_ptb_dataset, batch_size=batch_size,
                              shuffle=True)

val_ptb_dataset= PTBXL_Beat_DataSet(path=myPath, target_list=validList,
                                    leads=ecg_leads, upSample=upSample,
                                    pre=pre_R, post=post_R,myPad=padding,
                                    allowOverlap=overlap)
val_dataloader = DataLoader(val_ptb_dataset, batch_size=batch_size,
                            shuffle=True)

t_dataset = timing.tac(name='dataSetCreate')
print('Creating the training and validation datasets took %2i min %2i sec'
      %(t_dataset/60,t_dataset%60))

timing.tic(name='training')
for epoch in range(num_epochs):
    
    timing.tic(name='epoch')
    t_ep = 0 
    
    # Check for early stopping
    if early_stop_counter >= early_stop_limit-1 and epoch>=gracePeriod:
        if best_model_state:
            break
        else:
            best_model_state = deepcopy(model.state_dict())
            break
    
    ### Training
    # resample training data and set up data loader
    resample(train_ptb_dataset,upSample)
    train_dataloader = DataLoader(train_ptb_dataset, batch_size=batch_size,
                                  shuffle=True, worker_init_fn=seed_worker,
                                  generator=g)

    # Training loop: Forward -> Backprob -> Update params
    model.train()
    # reset variables
    train_preds, train_targs = [], [] 
    sample, train_features, train_labels = 0, 0, 0
    output, preds = 0, 0
    X_batch, y_batch, batch_loss = 0, 0, 0
    cur_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        # print(i)
        ## Get training data
        sample = data
        train_features, train_labels = sample['X'], sample['label']
        X_batch = get_variable(train_features).float()                         
        # Converts tensor to cuda
        
        ## Prediction
        output = model(X_batch)
        
        ## Eval training data
        preds = torch.max(output, 1)[1]
        train_targs += list(train_labels)
        train_preds += list(get_numpy(preds.data))                             
        # Converts tensor to NumPy
        
        ## compute gradients given loss
        y_batch = get_variable(train_labels).long()                            
        # Converts tensor to cuda
        batch_loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        cur_loss += batch_loss   
    train_loss.append(cur_loss / batch_size)
    train_acc_cur = balanced_accuracy_score(train_targs, train_preds)
    train_acc.append(train_acc_cur)
      
    ### Validation
    # Resample validation data and set up data loader
    resample(val_ptb_dataset,upSample)
    val_dataloader = DataLoader(val_ptb_dataset, batch_size=batch_size,
                                shuffle=True, worker_init_fn=seed_worker,
                                generator=g)
    
    ## Validation loop
    model.eval()
    # reset variables
    val_preds, val_targs = [], []
    sample, val_features, val_labels = 0, 0, 0
    X_batch = 0 
    output, preds = 0, 0
    valid_acc_cur = 0
    for i, data in enumerate(val_dataloader, 0):
        sample = data
        val_features, val_labels = sample['X'], sample['label']
        X_batch = get_variable(val_features).float()                           
        # Converts tensor to cuda
        
        output = model(X_batch)
        preds = torch.max(output, 1)[1]
        
        val_targs += list(val_labels)
        val_preds += list(get_numpy(preds.data))                               
        # Converts tensor to Numpy   
    
    valid_acc_cur = balanced_accuracy_score(val_targs, val_preds)
    ## Early stopping
    if early_stop and len(valid_acc):
        # if valid_acc_cur < valid_acc[-1]:
        if valid_acc_cur <= max(valid_acc):
            early_stop_counter += 1
        else:
            # torch.save((model.state_dict(), optimizer.state_dict()), path)
            best_model_state = deepcopy(model.state_dict())
            early_stop_counter = 0
            
    if epoch == 0:
        best_model_state = deepcopy(model.state_dict())
        
    valid_acc.append(valid_acc_cur)
    t_ep = timing.tac(name='epoch')
    
    if epoch % 1 == 0:
        # if epoch % 50 ==0:
        #     print(torch.cuda.memory_summary())
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f, %2i min %2i sec., stop counter: %1i" % (
                epoch+1, train_loss[-1], train_acc_cur, valid_acc_cur,
                t_ep/60,t_ep%60, early_stop_counter))
        # if early_stop_counter == 0:
        #     print('Saved a better model')

# Save the parameters of the best performing model
torch.save(best_model_state, model_path+model_id+'.pth')
        
epoch = np.arange(len(train_acc))
t_tr = timing.tac(name='training')
print('Training and validation took %2i min %2i sec'%(t_tr/60,t_tr%60))
 
#%% 7 - Testing
timing.tic(name='test')

## Load best model
model,_,_ = selectModel(netSelect, myPath, nr_leads, nr_classes, ecg_len)
model.load_state_dict(torch.load(model_path+model_id+'.pth'))
model.to(device)
model.eval()

## Make testing dataset and dataloader
test_ptb_dataset= PTBXL_DataSet(path= myPath, target_list=testList,
                                leads=ecg_leads, upSample='No')
test_dataloader = DataLoader(test_ptb_dataset,shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=g) # batch_size=1,

# Init for beatwise classification
true_beat = []
pred_beat = []
pred_beat_prob = []

# Init for full segment
true_seg = []
pred_seg_prob = []

# Init for different voting schemes
pred_maj = [] 
pred_avg = []
pred_one = []

softMax = Softmax(dim=1)
for i, data in enumerate(test_dataloader, 0):  
        # Get data          
        sample = data
        test_features, test_labels = sample['X'], sample['label']
        signal = get_numpy(test_features)      
        
        # Find beats
        beat_info = findBeatInfo(signal[0,0,:],pre=pre_R)                      
        # [r-peak, beat_start, beat_end]
        if beat_info == 'Error':
            continue
              
        # Loop through heartbeats, and make a prediction for each beat
        out_seg = []
        out_seg_prob = []
        pred_seg = []
        for row_nr, test_beat in enumerate(beat_info):
            # extract heartbeat
            X_beat = getBeat(signal, test_beat, ecg_len, nr_leads,
                             myPad=padding, allowOverlap=overlap)
            
            # Shape the heartbeat and make it into a tensor and pass it to cuda
            X_beat = X_beat.reshape(test_features.shape[0],
                                    test_features.shape[1],
                                    X_beat.shape[1])
            X_beat = torch.from_numpy(X_beat)         
            X_beat = get_variable(X_beat).float()                              
            # Converts tensor to cuda
            
            # Model output for heartbeat
            output = model(X_beat)
            output_prob = softMax(output)
            
            # True label of heartbeat
            true_beat += [int(test_labels)]
            
            # Make prediction from output
            pred_beat += [int(np.argmax(get_numpy(output), axis=1))]
            pred_beat_prob += [get_numpy(output_prob)[0,-1]]
            
            # Accumulate beatwise outputs for segment
            pred_seg += [pred_beat[-1]]
            out_seg += list(get_numpy(output))
            out_seg_prob += [get_numpy(output_prob)[0,-1]]
            
        # True label of segment
        true_seg += [int(test_labels)]
        
        # segment probaility
        pred_seg_prob += [np.mean(out_seg_prob,axis=0)]
        
        # majority voting
        pred_maj += [int(np.bincount(pred_seg).argmax())]
        
        # Mean value voting
        pred_avg += [int(np.argmax(np.mean(out_seg,axis=0), axis=0))] 

        # One-shot voting
        if len(classList)==2 and np.bincount(pred_seg)[0]>0:
            preds_one = 0
        elif len(classList)==2:
            preds_one = 1
        elif sum(np.bincount(pred_seg)[:-1])>0:
            preds_one = int(np.bincount(pred_seg).argmax())
        else:
            preds_one = len(classList)-1
            
        pred_one += [preds_one]   

t_test = timing.tac(name='test')
print('Testing took %2i min %2i sec'%(t_test/60,t_test%60))
#%% 8 - Results

sb = 0 # parameter to split the barplot according to classes
## Confusion Matrix, Accuracy measures
### Beatwise classification
conf_beat =confMatrix(true_beat, pred_beat ,classList,
                      figPath=model_path+'conf_beat.png', myTitle="Beats")
results_beat =  acc_measures(conf_beat,true_beat, pred_beat ,
                             figPath=model_path+'res_beat.png',plotIt=1,
                             splitBars=sb,title='Beatwise classification')

### Segmentwise classification
# Majority voting
conf_maj =confMatrix(true_seg, pred_maj ,classList,
                     figPath=model_path+'conf_major.png',myTitle="Majority")
results_maj =  acc_measures(conf_maj, true_seg, pred_maj ,
                            figPath=model_path+'res_maj.png',plotIt=1,
                            splitBars=sb,title='Majority voting')

# # Average voting
# conf_avg =confMatrix(true_seg, pred_avg ,classList,
#                      figPath=model_path+'conf_avg.png',myTitle="Average")
# results_avg =  acc_measures(conf_avg, true_seg, pred_avg ,
#                             figPath=model_path+'res_avg.png',plotIt=1,
#                             splitBars=sb,title='Average voting')

# # At least one voting
# conf_one =confMatrix(true_seg, pred_one ,classList,
#                      figPath=model_path+'conf_one.png',myTitle="One-shot")
# results_one =  acc_measures(conf_one, true_seg, pred_one ,
#                             figPath=model_path+'res_one.png',plotIt=1,
#                             splitBars=sb,title='One-shot voting')

## ROC plot - only for binary cases
if len(classList)==2:
    roc_beat = roc_auc(true_beat, pred_beat_prob,plotIt=0)
    roc_seg = roc_auc(true_seg, pred_seg_prob,plotIt=0)    
    rocList=[roc_beat]
    mylegend=['Segment AUC: %0.2f' %(roc_beat[-1])]  
    # rocList=[roc_beat,roc_seg]
    # mylegend=['Beatwise AUC: %0.2f' %(roc_beat[-1]),'Segment AUC: %0.2f' 
    #           %(roc_seg[-1])]  
    compare_roc(rocList,mylegend,figPath=model_path+'roc.png')

table,total_params = count_parameters(model)
#%% 9 - Print learning curve 
learningCurve(epoch, train_acc, valid_acc,figPath=model_path)

#%% 10 - Save logs
comment = 'Deep'
with open(model_path+'info.txt', 'w') as f:
    f.write('Model: '+ modelDict[netSelect])
    f.write('\nLeads used: '+str(ecg_leads))
    f.write('\nClasses: '+str(classList))
    f.write('\n Exclude list: '+str(excludeList))
    f.write('\nTrainable params:'+str(total_params))
    f.write('\nSignal length: ' + str(ecg_len))
    f.write('\n')
    f.write('\nTraining hyperparameters\n')
    for key, value in training_params.items():  
        f.write('- %s: %s\n' % (key, value))
    
    f.write('\nNetwork parameters\n')
    for key, value in network_params.items():  
        f.write('- %s: %s\n' % (key, value))
        
    f.write('\n')
    f.write('Training and validation took %2i min %2i sec'%(t_tr/60,t_tr%60))
    f.write("\nLast epoch was: Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f, %2i min %2i sec., stop counter: %1i" % (
                len(epoch), train_loss[-1], train_acc_cur, valid_acc_cur,
                t_ep/60,t_ep%60, early_stop_counter))

    f.write('\nResults: \n'+str(results_beat))
    # f.write(comment)
print('Script is finished, logs are saved')