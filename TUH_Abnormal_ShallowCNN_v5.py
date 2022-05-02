#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install torch
#!pip install mne
#!pip install braindecode


# In[4]:


import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import mne
# import logging

from braindecode.datasets.tuh import TUHAbnormal, TUH
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)


plt.style.use('seaborn')
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted


# In[2]:**Loading the data . . .**


TUH_PATH = 'C:/Users/Kitti/Documents/Thesis/TUH/Abnormal/SSample/' # specify the path to the TUH Abnormal dataset
N_JOBS = 2  # specify the number of jobs for loading and windowing
tuh = TUHAbnormal(
    path=TUH_PATH,
    recording_ids=None,
    target_name='pathological',
    preload=False,
    add_physician_reports=True,
    n_jobs=1 if TUH.__name__ == '_TUHMock' else N_JOBS,  # Mock dataset can't
    # be loaded in parallel
)


# In[4]:


# This function discards all the channels found in the recordings that have
# an incomplete configuration, and keep only those channels that we are interested in, i.e. the 21
# channels of the international 10-20-placement). The dataset is subdivided into
# recordings with 'le' and 'ar' reference which we will have to consider.

short_ch_names = sorted([
    'A1', 'A2',
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'])
ar_ch_names = sorted([
    'EEG A1-REF', 'EEG A2-REF',
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
le_ch_names = sorted([
    'EEG A1-LE', 'EEG A2-LE',
    'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
    'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
    'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
    'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'])
assert len(short_ch_names) == len(ar_ch_names) == len(le_ch_names)
ar_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
    ar_ch_names, short_ch_names)}
le_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
    le_ch_names, short_ch_names)}
ch_mapping = {'ar': ar_ch_mapping, 'le': le_ch_mapping}


def select_by_channels(ds, ch_mapping):
    split_ids = []
    for i, d in enumerate(ds.datasets):
        ref = 'ar' if d.raw.ch_names[0].endswith('-REF') else 'le'
        # these are the channels we are looking for
        seta = set(ch_mapping[ref].keys())
        # these are the channels of the recoding
        setb = set(d.raw.ch_names)
        # if recording contains all channels we are looking for, include it
        if seta.issubset(setb):
            split_ids.append(i)
    return ds.split(split_ids)['0']

tuh = select_by_channels(tuh, ch_mapping)


# In[7]:


# Based on: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.crop

def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


# Extra Custom function for renaming channels

# In[6]:


def custom_rename_channels(raw, mapping):
    # rename channels which are dependent on referencing:
    # le: EEG 01-LE, ar: EEG 01-REF
    # mne fails if the mapping contains channels as keys that are not present
    # in the raw
    reference = raw.ch_names[0].split('-')[-1].lower()
    assert reference in ['le', 'ref'], 'unexpected referencing'
    reference = 'le' if reference == 'le' else 'ar'
    raw.rename_channels(mapping[reference])



# In[8]:


# **Preprocessor Pipeline**
# 
# The **4th** and **5th** steps of the preprocessing, the **clipping at $\pm800\mu$v** and the **resampling** of the records don't have separate functions written, as we are using the mne package Raw class's functions.

# In[7]:


# parameters to be defined for the preprocessing pipeline
tmin = 1 * 60
tmax = 17 * 60 # as the first minute of each recording is cropped, n+1 minutes must be added
sfreq = 100

preprocessors = [
    Preprocessor(custom_crop, tmin=tmin, tmax=tmax, include_tmax=True,
                 apply_on_array=False),
    Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'), # mne Raw class function
    Preprocessor(custom_rename_channels, mapping=ch_mapping, # rename channels to short channel names
                 apply_on_array=False), #
    Preprocessor('pick_channels', ch_names=short_ch_names, ordered=True), # mne Raw class function
    Preprocessor(multiply, factor=1e6, apply_on_array=True), # scaling signals to microvolt
    Preprocessor(np.clip, a_min=-800, a_max=800, apply_on_array=True), # clip outlier values to +/- 800 micro volts
    Preprocessor('resample', sfreq=sfreq), # mne Raw class function
]



# In[9]:
'''
import os.path
# Create output folder
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'Output2')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
OUT_PATH = final_directory  # please insert actual output directory here
'''

N_JOBS = 2 # the number of CPUs to be used

tuh_preproc = preprocess(
    concat_ds=tuh,
    preprocessors=preprocessors,
    n_jobs=N_JOBS,
    save_dir=None
)


# In[13]: Train-test split
splits = tuh_preproc.split("train")
tuh_train, tuh_test = splits['True'], splits['False']

# Further splitting the train set into train and valid sets (90-10% ratio)
#%% Train-valid split

train_len = len(tuh_train.datasets)
train_size = 0.9
valid_size=0.1
tuh_train_inds = [*range(train_len)]
train_index = int(train_len*train_size)-1
new_train_inds = [*range(train_index)]


new_val_inds = list(set(tuh_train_inds)-set(new_train_inds))
split_inds = [new_train_inds, new_val_inds]

#%%
splits = tuh_train.split(split_ids=split_inds)
tuh_train, tuh_valid = splits['0'],splits['1']

# In[14]: **Building and Training a basic CNN model**


import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
# to_dense_prediction_model,

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 2
# Extract number of chans and time steps from dataset
in_chans, input_size_samples = tuh_train[0][0].shape


model = ShallowFBCSPNet(
    in_chans,
    n_classes,
    input_window_samples=6000,
    final_conv_length=25,
)


# Send model to GPU
if cuda:
    model.to(device)



# In[167]:

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
from braindecode.models import get_output_shape

input_window_samples =6000
n_preds_per_input = get_output_shape(model, in_chans, input_window_samples)[2] #

train_set = create_fixed_length_windows(
    tuh_train,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=False,
    mapping={False: 0, True: 1},  # map non-digit targets
)

valid_set = create_fixed_length_windows(
    tuh_valid,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=False,
    mapping={False: 0, True: 1},  # map non-digit targets
)

test_set = create_fixed_length_windows(
    tuh_test,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=False,
    mapping={False: 0, True: 1},  # map non-digit targets
)

#%% Target transform

train_set.target_transform = lambda x: np.full((n_preds_per_input), fill_value=x)
valid_set.target_transform = lambda x: np.full((n_preds_per_input), fill_value=x)
test_set.target_transform = lambda x: np.full((n_preds_per_input), fill_value=x)
    

#%% Create train DataLoader
from torch.utils.data import DataLoader
batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# In[14]:

from torch import optim
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier
# import torch.nn.functional as F
from skorch.helper import predefined_split

# hyperparameters for training the model
lr = 1e-3
optimizer = optim.Adam(params=model.parameters(), lr=lr)
criterion = torch.nn.NLLLoss()
n_epochs = 3  # we use few epochs for speed and but more than one for plotting
max_epochs = 35


#%% CNN training with iterator function
# Inspo for more: https://www.kaggle.com/code/kdnishanth/pytorch-cnn-tutorial-in-gpu/notebook
from datetime import datetime


patience = 2
print_every = 5

def train(model, device, criterion, optimizer, train_loader, valid_loader, epochs=5, early_stop_patience=2):
    
    model.train()
    the_last_loss = 100
    trigger_times = 0
    train_losses_per_epoch, valid_losses_per_epoch = [], []
    
    # modify this later and find a nicer solution
    for i in range(1, 10000):
        if len(train_loader) / i <= 20:
            print_every = i
            break
        
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        correct = 0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, inds = data
            inputs = inputs.float()
            labels = torch.clone(labels.long())
            
            # sending input to GPU/CPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.numel()
            correct += predicted.eq(labels).sum().item()
          
            # Statistics
            running_loss += loss.item()
            train_acc = correct/total
            running_acc += train_acc
            
            
            if i % print_every == print_every-1 or i+1 == len(train_loader):
                now = datetime.now()
                current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                print('Time: {}. . [{}/{}, {}/{}] train_loss: {:.6}, train_acc: {:.3}'.format(current_time, epoch+1, n_epochs, i+1, len(train_loader), running_loss/(i+1), running_acc/(i+1)))
    
        train_losses_per_epoch.append(running_loss/len(train_loader))
        current_loss = validation(model, device, valid_loader, criterion)
        print('The Current Validation Loss:', current_loss)
            
        if current_loss > the_last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
    
            if trigger_times >= early_stop_patience:
                print('Early Stopping!\nStart the test process.')
                return model, train_losses_per_epoch, valid_losses_per_epoch
                # this needs to be modified to start the testing process -> return model if turn into a function
    
        else:
            print('Trigger Times: 0')
            trigger_times = 0
    
        the_last_loss = current_loss
        valid_losses_per_epoch.append(current_loss)
    
    print('Finished Training')        
    return model, train_losses_per_epoch, valid_losses_per_epoch
    
    
 #%% Validation function

def validation(model, device, valid_loader, loss_function):

     model.eval()
     loss_total = 0

     # Test validation data
     with torch.no_grad():
         for data in valid_loader:
             inputs, labels, inds = data
             inputs = inputs.float()
             labels = torch.clone(labels.long())
             
             #Sending input to GPU/CPU
             inputs, labels = inputs.to(device),  labels.to(device)
             

             output = model(inputs)
             loss = loss_function(output, labels)
             loss_total += loss.item()

     return loss_total / len(valid_loader)  

    
# Test function

def test(device, model, test_loader):
    
    model.eval()
    total = 0
    correct = 0
    accuracy = 0
    labels_all = torch.tensor([])
    predicted_all = torch.tensor([])
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, inds = data
            inputs = inputs.float()
            labels = torch.clone(labels.long())

            output = model(input)
            _, predicted = torch.max(output.data, 1)

            total += labels.numel()
            correct += (predicted == labels).sum().item()
            
            
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals =  top_class == labels.view(*top_class.shape)
            accuracy +=   torch.mean(equals.type(torch.FloatTensor)).item()
            predicted_all = torch.cat((predicted_all, predicted) ,dim=0)
            labels_all = torch.cat((labels_all, labels) ,dim=0)
            

    print('Accuracy: {:.2}'.format(correct/total))
    print('Accuracy2: {:.2}'.format(accuracy/len(test_loader)))
    return labels_all, predicted_all
          
          
    
#%%Test train function

trained_model, train_losses, valid_losses = train(model, device, criterion, optimizer, train_loader, valid_loader, n_epochs)

#%% Testing the test function
y_true, y_pred = test(device, trained_model, test_loader)


#%%
# Save the trained model

PATH = './cnnshallow.pth'
torch.save(model.state_dict(), PATH)

#how to reload the saved model:
'''
model = ShallowFBCSPNet(
    in_chans,
    n_classes,
    input_window_samples=1000,
    final_conv_length=1,
)
model.load_state_dict(torch.load(PATH))
'''

#%% Plot Training loss curve

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

#%% Evaluation metrics

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
cl_report = classification_report(y_true.flatten(), y_pred.flatten())

#%% Plot confusion matrix
ax = sns.heatmap(conf_matrix, annot=True,  fmt = 'd', cmap='Blues')

ax.set_title('Confusion Matrix');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

#%% Plot confusion matrix with percentages
ax1 = sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, fmt='.2%', cmap='Blues')

ax1.set_title('Confusion Matrix');
ax1.set_xlabel('\nPredicted Values')
ax1.set_ylabel('Actual Values');

## Ticket labels - List must be in alphabetical order
ax1.xaxis.set_ticklabels(['False','True'])
ax1.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()





