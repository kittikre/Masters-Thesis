#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# **Install packages . . .**


# In[1]: **Import packages . . .**



# import tempfile

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import mne
# import logging

from braindecode.datasets.tuh import TUHAbnormal, TUH
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)

# from torch.utils.data import DataLoader


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


# In[3]:


print(tuh.description.head())


# ##### 1. Function for selecting a subset of 21 electrode positions following the international 10-20 placement
# ![image.png](attachment:image.png)
# 
# This function is not part of the preprocessing pipeline, hence needs to be executed separately.

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


# ##### 2-3. Function for cropping the first n seconds of every recording & use a maximum of m seconds
# 
# This function is part of the preprocessing pipeline.

# In[5]:


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


# **Preprocessor Pipeline**
# 
# The **4th** and **5th** steps of the preprocessing, the **clipping at $\pm800\mu$v** and the **resampling** of the records don't have separate functions written, as we are using the mne package Raw class's functions.

# In[7]:


# parameters to be defined for the preprocessing pipeline
tmin = 1 * 60
tmax = 21 * 60 # as the first minute of each recording is cropped, this is how we can keep 20 minutes of the recordings
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


import os.path

# Create output folder
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'Output')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

N_JOBS = 2 # the number of CPUs to be used
OUT_PATH = 'C:/Users/Kitti/Documents/Python Scripts/Output/'  # please insert actual output directory here
tuh_preproc = preprocess(
    concat_ds=tuh,
    preprocessors=preprocessors,
    n_jobs=N_JOBS,
    save_dir=OUT_PATH
)


# **A look into the preprocessed dataset**
# 
# Two newly added columns can also be found in the preprocessed dataset, *channels* and *length*, with the purpose of verifying the accuracy of the preprocessing.

# In[10]:


# Adding channel positions for each recording
tuh_extended = tuh_preproc.description
channel_names = [] # np.empty(shape=(2993,))
for i in range(len(tuh_preproc.datasets)): 
    channel_names.append(tuh_preproc.datasets[i].raw.ch_names)
tuh_extended['channels'] = channel_names

# Adding the cropped length of each recording
length = [] # np.empty(shape=(2993,))
for i in range(len(tuh_preproc.datasets)): 
    length.append(tuh_preproc.datasets[i].raw.n_times / tuh_preproc.datasets[i].raw.info['sfreq'])
tuh_extended['length'] = length


# One-hot encoding for labels . . .

# **Split preprocessed dataset to train and eval sets**

# In[71]:

'''
# Create windows
tuh_windows = create_fixed_length_windows(
    tuh_preproc,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=1000,
    window_stride_samples=1000,
    drop_last_window=False,
    mapping={False: 0, True: 1},  # map non-digit targets
)
# store the number of windows required for loading later on
tuh_windows.set_description({
    "n_windows": [len(d) for d in tuh_windows.datasets]})
'''

# In[13]: Train-test split


splits = tuh_preproc.split("train")
tuh_train, tuh_eval = splits['True'], splits['False']


# In[14]: **Building and Training a basic CNN model**


import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from braindecode.models import get_output_shape
# to_dense_prediction_model,

# In[ ]:


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

n_classes = 1
# Extract number of chans and time steps from dataset
in_chans, input_size_samples = tuh_train[0][0].shape


model = ShallowFBCSPNet(
    in_chans,
    n_classes,
    input_window_samples=1000,
    final_conv_length=2,
)


# Send model to GPU
if cuda:
    model.cuda()


# In[167]:


# from braindecode.preprocessing import create_fixed_length_windows

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.

input_window_samples =1000
n_preds_per_input = get_output_shape(model, in_chans, input_window_samples)[2] # =887

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

'''
valid_set = create_fixed_length_windows(
    tuh_eval,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=False,
    mapping={False: 0, True: 1},  # map non-digit targets
)


test_set = create_fixed_length_windows(
    test_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    targets_from='channels',
    last_target_only=False,
    preload=False
)
'''



#%% Target transform

train_set.target_transform = lambda x: np.full((1, 60), x)

# Turning X and y into nparrays . . .
# In[193]:


train_set_X = []
for x in train_set:
    train_set_X.append(np.asarray(x[0], dtype=np.float32))
train_set_X = np.asarray(train_set_X, dtype=np.float32)


# In[189]:


train_set_y = []
for x in train_set:
    train_set_y.append(x[1])
train_set_y = np.asarray(train_set_y, dtype=np.int32)



# In[212]:


# from skorch.helper import predefined_split
from braindecode.training import TimeSeriesLoss
from torch import optim
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier
import torch.nn.functional as F

# hyperparameters for training the model
lr = 1e-3
optimizer = optim.Adam
batch_size = 64
n_epochs = 4  # we use few epochs for speed and but more than one for plotting
max_epochs = 35

from sklearn.metrics import balanced_accuracy_score

def balanced_accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    return balanced_accuracy_score(y.flatten(), y_pred.flatten())


train_bal_acc = EpochScoring(
    scoring=balanced_accuracy_multi,
    on_train=True,
    name='train_bal_acc',
    lower_is_better=False,
)
valid_bal_acc = EpochScoring(
    scoring=balanced_accuracy_multi,
    on_train=False,
    name='valid_bal_acc',
    lower_is_better=False,
)
callbacks = [
    ('train_bal_acc', train_bal_acc),
    ('valid_bal_acc', valid_bal_acc)
]

#%%

clf = EEGClassifier(
    model,
    optimizer=optimizer,
    criterion=TimeSeriesLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    iterator_train__shuffle=False,
    # iterator_train=train_set,
    # iterator_valid=valid_set,
    train_split=None,
    # train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device,
)

#%%
# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(train_set_copy, y=None, epochs=n_epochs)



# In[124]:


import braindecodeold as braindecodeold
from braindecodeold.experiments.experiment import Experiment


# In[ ]:


'''
To-do:
- Go through the code below and identify/modify any classes/functions that needs to be imported in Spyder,
- See if there aren't any missing parameters left
- Try to run the experiment

'''
cuda = False
iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
optimizer = optim.Adam(model.parameters(), lr=init_lr)

loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)

if model_constraint is not None:
    assert model_constraint == 'defaultnorm'
    model_constraint = MaxNormDefaultConstraint()
monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'), CroppedDiagnosisMonitor(input_time_length, n_preds_per_input), RuntimeMonitor(),]
stop_criterion = MaxEpochs(max_epochs)
batch_modifier = None
run_after_early_stop = True
exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop,
                     batch_modifier=batch_modifier,
                     cuda=cuda)
exp.run()

