# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:24:39 2021
Data functions for handling the PTB-XL dataset
Each method has its own description in it's header section.'
The methods defined in this file are:
    sortClassLabel()
    listISs_ofClass()
    listIDs_Rest()
    splitClassList()
    getTargets()
    
The classes defined in this file are:
    PTBXL_Beat_DataSet
    PTBXL_DataSet
    

@author: Vespa
"""
import numpy as np
import pandas as pd
import ast
import json
import wfdb

from biosppy.signals import ecg
from Methods.plotECG import plotECG

np.random.seed(8)

#%% Functions to prepare class lists
def sortClassLabel(rawLabels , translate_dict): 
    """
    Translates the raw labels into the desired labels 
    Helper function for listIDs_ofClass
    
    input: rawLabels - the labels to translate
           tranlate_dict - a dictionary of the label mapping          
    output: list of translated labels    
    """
    # This can handle multiple output classes as well
    out = []
    for row in range(0,len(rawLabels)):
        tmp = []
        y_dic = rawLabels.iloc[row]
        for key in y_dic.keys():
            if key in translate_dict.index:
                tmp.append(translate_dict.loc[key])
        out.append(list(set(tmp)))
    return out

def listIDs_ofClass(path, className, subclass=True, save=False,
                    exclude_list=[['Nan'],['Nan']]):
    """
    Makes a list containing the ecg_ID-s of the recordings belonging to 
    the given class.
    It is a helper function for getTargets
    
    Example inputs:
        path =  the right path to the file folder containing the csv'
        classname ='AMI'
        subclass = True
        save = True
        exclude_list=[['MI'],['PMI']]
    Output:
        List of ecg_ID-s belonging to desired class
    """
    # Load annotation from 'ptbxl_database.csv
    Y_raw = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    labels_raw = Y_raw.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load diagnostic aggregationfrom 'scp_statements.csv
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)   
    
    # Make a dictionary connecting diagnostic labels to class
    if subclass:
        myDict = agg_df[agg_df.diagnostic_subclass == className].diagnostic_subclass
    else:
        myDict = agg_df[agg_df.diagnostic_class == className].diagnostic_class

    # Sort recordings with the class label
    myLabels = sortClassLabel(labels_raw, myDict)                               
    
    # Only keep the ecg_ID-s belonging to this class
    myList = list(pd.DataFrame(myLabels, index = Y_raw.index).dropna().index)
    
    # Exclude sub classesgiven in exclude_list
    for _,cat in enumerate(exclude_list[0]):
        if cat == className:
            for _,ex in enumerate(exclude_list[1]):
                exDict = agg_df[agg_df.diagnostic_subclass == ex].diagnostic_subclass
                if len(exDict)<1:
                    print(f'`{ex}` is not a subclass, so it is skipped')
                    break
                exLabels = sortClassLabel(labels_raw, exDict)
                exList = list(pd.DataFrame(exLabels, index = Y_raw.index).dropna().index)
                for item in exList :
                    try: myList.remove(item)
                    except: continue 
                # Possible error: item has 2 labels and based on the first one 
                # it has already ben excluded     
    
    ## Save lists to CSV
    if save:
        np.savetxt(path+'list_'+className+'.csv', myList, delimiter =', ',
                   fmt ='% s')
    
    return myList
    
def listIDs_Rest(path,listOfClasses):
    """
    Make the list containing the ecg_ID-s of all recordings that are not in
    any specific class (ID is not in listOfClasses)
    It is similar to listIDs_ofClass.
    It is a helper function used in getTargets
    
    Inputs:
        path
        listOfClasses
    Outputs:
        list of ecg_IDs belonging to recordings not in class list  
    """
    Y_raw = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    All = list(Y_raw.index)
    list_Other = [All[i] for i in range(len(All)) if All[i] not in (listOfClasses)]
    
    return list_Other

def splitClassList(path, myList, paramsJson=False):
    """ 
    Splits the recordings from the input list into training, validation and
    test sets according to the predefined distribution by the authors of the
    PTB-XL database publication.
    This method is called in the getTargets function.
    Inputs:
        path
        myList - list of ecg-IDs to split
        paramsJson - optionally it can take the split rules from a json file
    Outputs:
        trainList, validList, testList      
    """
    Y_raw = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    fold = Y_raw.strat_fold
   
    # Split parameters
    if paramsJson == False:
        train = [1,2,3,4,5,6,7,8]
        test = [10]
        validation = [9]
    else:
        with open(path+'json\\'+paramsJson, 'r') as jfile:
            split_params = json.load(jfile)
        train = split_params['train']
        test  = split_params['test']
        validation = split_params['validation']
    # Split
    trainList = [myList[i] for i in range(len(myList)) if fold.loc[myList[i]] in train]
    validList = [myList[i] for i in range(len(myList)) if fold.loc[myList[i]] in validation]
    testList  = [myList[i] for i in range(len(myList)) if fold.loc[myList[i]] in test]
    
    return trainList, validList, testList

def getTargets(path, classes, subclass=True,exclude_list=[['Nan'],['Nan']]):
    """ 
    Makes a list of the ecg_ids of the desired classes.
    Inputs:
        path
        classes - a list of classes to use
        subclasses - it gives information on the values in classes, 
                     if they are names of classes or subclasses.
        exclude_list - list of classes to exclude, first element is the list 
                    of classes to excude from, second is the list of 
                    subclasses to exclude.
    Outputs:
        trainList, validList, testList
    """
    # Init variables
    myList = []
    flatClassList = []
    myTrainList = []
    myValidList = []
    myTestList = []
    
    skipFromOther = []
    if 'Other' in exclude_list[0]:
        skipFromOther += exclude_list[1]
    
    # Loop throug classes
    for c in range(len(classes)):
        if classes[c] == 'Other':
            # Take the ecg_id-s for the 'Other' class
            myList.append(listIDs_Rest(path, flatClassList+skipFromOther))
            # make the split
            tmpTrainList, tmpValidList, tmpTestList = splitClassList(path, myList[c],paramsJson=False)
        
            # append as a dimension to the list 
            myTrainList.append(tmpTrainList)
            myValidList.append(tmpValidList)
            myTestList.append(tmpTestList)
            continue
        # Get ecg_id-s for given class
        myList.append(listIDs_ofClass(path, classes[c], subclass=subclass, 
                                      save=False,exclude_list=exclude_list))
        # Make flat list for the 'Other' class             
        flatClassList += myList[c]
        
        # Make  the split of the data 
        tmpTrainList, tmpValidList, tmpTestList = splitClassList(path, myList[c],paramsJson=False)
        
        # By appending, it will save the list of ecg_id-s of a given class
        # as a dimension of a multi-dim list
        myTrainList.append(tmpTrainList)
        myValidList.append(tmpValidList)
        myTestList.append(tmpTestList)
    
    return myTrainList, myValidList, myTestList  


#%% Beats Dataset
from torch.utils.data import Dataset

class PTBXL_Beat_DataSet(Dataset):
    """ 
    This is a Dataset class for heartbeat segmented data.
    Inputs:
        path
        target_list - list of ecg_IDs for the given set
        upSample -variable that sets if upsampling, downsampling or no special
                    sampling strategy is performed
        leads - list of leads to include
        transform - no implemented functionality
        target_transform - no implemented functionality
        pre - start of a heartbeat before the R-peak in seconds
        post - end of a heartbeat after the R-peak in seconds
        myPad-padding strategy for the heartbeat, accepts same args as np.pad
        allowOverlap - sets if overlap in heartbeats is allowed
    """
    def __init__(self, path, target_list, upSample = False, leads= ['V5'],
                 transform=None, target_transform=None, pre=0.3, post=0.5,
                 myPad='edge',allowOverlap=0):
        # Settings
        # print("Init started")
        self.path = path
        self.upSample = upSample
        self.freq = 100
        self.pre = pre
        self.post = post
        self.myPad = myPad
        self.allowOverlap = allowOverlap
        
        self.leads = leads
        leads_list = ['I','II','III','aVL','aVR','aVF',
                      'V1','V2','V3','V4','V5','V6']
        self.lead_dict = dict(zip(leads_list,range(0,12)))
        if self.leads == ['all']:
            self.leads = leads_list
        
        self.transform = transform
        self.target_transform = target_transform
        
        # Raw unbalanced information about the dataset
        self.Y_raw = pd.read_csv(self.path+'ptbxl_database.csv',
                                 index_col='ecg_id')
        self.fileNames = self.Y_raw['filename_lr'].copy()
        
        # Get beats
        myBeats = []
        for target_class,id_list in enumerate(target_list):
            # print('start class '+str(target_class))
            myBeats.append(self.getRpeaks(id_list))
        # self.ecg_labels = target_list
        self.ecg_labels = myBeats
        
        # Processed data information with balanced classes
        self.dataInfo = None
        # self.labelInfo = None       
        if upSample == 1:
            self.upSampleMinor()
        elif upSample == 0:
            self.downSampleMajor()
        else:
            self.noSampling()
            print('No up or downsamplling is performed')

        # print("end of init")
        
    def __len__(self):
        # return len(self.ecg_labels)
        return len(self.dataInfo)

    def __getitem__(self, idx):
        # Load raw signal data
        X_full = [wfdb.rdsamp(self.path+ self.dataInfo['filename'].iloc[idx])]
        X_full = np.array([signal for signal, meta in X_full]) 
        # shape: [1,1000,12]
        
        # Define beat boundaries
        start = int(self.dataInfo['beat_s'].iloc[idx])
        if self.allowOverlap:
            end = int(self.dataInfo['Rpeak'].iloc[idx])+int(self.post*self.freq)
        else:
            end = int(self.dataInfo['beat_e'].iloc[idx])
        beat_len = int((self.pre+ self.post)* self.freq)
        
        # Preallocate X
        X = np.zeros([len(self.leads),beat_len])
        # start = int(self.dataInfo['Rpeak'].iloc[idx] - self.freq*self.pre)

        # Truncate
        if (end-start) > beat_len:
            end = int(start+beat_len)
            for i,key in enumerate(self.leads):
                X[i] = X_full[0,start:end, self.lead_dict[key]]
        # Zero-pad -> pad with last value
        elif (end-start) < beat_len:
            nr_zeros = beat_len-(end-start)
            for i,key in enumerate(self.leads):
                X[i] = np.pad(X_full[0,start:end, self.lead_dict[key]],
                              (0,nr_zeros),mode=self.myPad)
                # pad with the last value
        # Do nothing, size is perfect
        else:
            for i,key in enumerate(self.leads):
                X[i] = X_full[0,start:end, self.lead_dict[key]]      
        
        X = X.reshape([len(self.leads) , X.shape[1]])
        # print('succes')
        # label = self.ecg_labels['class'].iloc[idx,-1]
        label = self.dataInfo['class'].iloc[idx]
        
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"X": X, "label": label}
    
        return sample
        
    
    def downSampleMajor(self):
        # print("Downsampling majority class")
        # Clear the lists
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              'filename','Rpeak', 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples=(sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        n_samples = min(target_shape)
        for c in range(len(target_shape)):           
            # make random idx list
            idx_rand = np.random.permutation(len(self.ecg_labels[c]))
            
            # Pick n_samples random samples to keep 
            # samples_idx = [self.ecg_labels[c][i] for i in idx_rand[0:self.n_samples]]
            sample_ecg_id = [self.ecg_labels[c][i][0] for i in idx_rand[0:n_samples]]
            sample_rpeak  = [self.ecg_labels[c][i][1] for i in idx_rand[0:n_samples]] 
            sample_beat_s = [self.ecg_labels[c][i][2] for i in idx_rand[0:n_samples]] 
            sample_beat_e = [self.ecg_labels[c][i][3] for i in idx_rand[0:n_samples]] 
            # Add that to the list
            # df = pd.DataFrame(self.fileNames[samples_idx])
            # df[self.fileNames.index.name] = df.index
            # df['class'] = np.ones(len(samples_idx))*c
            df = pd.DataFrame(columns=(self.fileNames.index.name, 'filename',
                                       'Rpeak','beat_s','beat_e', 'class'))
            df['filename'] = self.fileNames[sample_ecg_id]
            df['Rpeak']= sample_rpeak 
            df['beat_s']= sample_beat_s
            df['beat_e']= sample_beat_e
            
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(sample_ecg_id))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)
                    
        # # Shuffle
        # random_idx = np.random.permutation(len(self.dataInfo))
        # self.dataInfo = self.dataInfo.iloc[random_idx]
        
    def upSampleMinor(self):
        # Clear the lists
        # self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name, 
        #                                       self.fileNames.name, 'class'))
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              'filename','Rpeak', 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples=(sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        n_samples = max(target_shape)
        for c in range(len(target_shape)): 
            # make random idx list
            multi = int(np.ceil(n_samples / len(self.ecg_labels[c])))
            long_ecg_labels = []
            long_ecg_labels += self.ecg_labels[c]*multi  
            
            idx_rand = np.random.permutation(len(long_ecg_labels)) 
            # idx_rand = np.random.randint(low=0,high=len(self.ecg_labels[c]),
            #                               size=self.n_samples)
            
            # Pick n_samples random samples to keep
            # samples_idx=[long_ecg_labels[i] for i in idx_rand[0:self.n_samples]]
            sample_ecg_id = [long_ecg_labels[i][0] for i in idx_rand[0:n_samples]]
            sample_rpeak  = [long_ecg_labels[i][1] for i in idx_rand[0:n_samples]] 
            sample_beat_s = [long_ecg_labels[i][2] for i in idx_rand[0:n_samples]] 
            sample_beat_e = [long_ecg_labels[i][3] for i in idx_rand[0:n_samples]] 
            # Add that to the list
            # df = pd.DataFrame(self.fileNames[samples_idx])
            # df[self.fileNames.index.name] = df.index
            # df['class'] = np.ones(len(samples_idx))*c
            df = pd.DataFrame(columns=(self.fileNames.index.name, 'filename',
                                       'Rpeak','beat_s','beat_e', 'class'))
            df['filename'] = self.fileNames[sample_ecg_id]
            df['Rpeak']= sample_rpeak 
            df['beat_s']= sample_beat_s
            df['beat_e']= sample_beat_e
            
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(sample_ecg_id))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)
                    
        # # Shuffle
        # random_idx = np.random.permutation(len(self.dataInfo))
        # self.dataInfo = self.dataInfo.iloc[random_idx]
    def noSampling(self):
        # Clear the lists
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              'filename','Rpeak', 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples = (sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        # n_samples = min(target_shape)
        for c in range(len(target_shape)): 
            # n_samples = len(self.ecg_labels[c])
            # make random idx list
            idx_rand = np.random.permutation(len(self.ecg_labels[c]))
            
            # Pick n_samples random samples to keep 
            # samples_idx=[self.ecg_labels[c][i] for i in idx_rand[0:self.n_samples]]
            sample_ecg_id = [self.ecg_labels[c][i][0] for i in idx_rand]
            sample_rpeak  = [self.ecg_labels[c][i][1] for i in idx_rand] 
            sample_beat_s = [self.ecg_labels[c][i][2] for i in idx_rand] 
            sample_beat_e = [self.ecg_labels[c][i][3] for i in idx_rand] 
            # Add that to the list
            # df = pd.DataFrame(self.fileNames[samples_idx])
            # df[self.fileNames.index.name] = df.index
            # df['class'] = np.ones(len(samples_idx))*c
            df = pd.DataFrame(columns=(self.fileNames.index.name, 'filename',
                                       'Rpeak','beat_s','beat_e', 'class'))
            df['filename'] = self.fileNames[sample_ecg_id]
            df['Rpeak']= sample_rpeak 
            df['beat_s']= sample_beat_s
            df['beat_e']= sample_beat_e
            
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(sample_ecg_id))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)
    def getRpeaks(self, input_list, plotIt=0, flatten=1):
        pre_s = self.pre*self.freq
        rpeaks_list = []
        self.error_idx = []
        for idx, val in enumerate(input_list):
            # Get X_raw
            X_raw = [wfdb.rdsamp(self.path+self.fileNames.loc[val])]
            X_raw = np.array([signal for signal, meta in X_raw]) 
            # shape: [1,1000,12]
            
            signal = X_raw[0,:,self.lead_dict[self.leads[0]]]
            
            # Get rpeaks
            try:
                _,_,rpeaks,_,_,_,_ = ecg.ecg(signal,self.freq,show=False)
                # rpeaks = ecg.christov_segmenter(signal=signal, sampling_rate=self.freq)
                # print('christov did it')
            except:
                self.error_idx += [val]
                print('Recording with ecg_id '+str(val)+' was skipped due to an error')
                continue
            
            # Visualize
            if plotIt:
                plotECG(signal ,scale = 2.5, title = 'R-peaks owerlayed',
                        rpeaks=rpeaks/self.freq)
            
            # Save rpeaks into a list of lists: rpeaks_list = [ecg_id, rpeak, start, end]
            if flatten:
                rp_list = rpeaks.tolist()
                last_rp = len(rp_list)
                [rpeaks_list.append([input_list[idx],int(rp_list[rp_idx]),
                                     int(rp_list[rp_idx]-pre_s),
                                     int(rp_list[rp_idx+1]-pre_s)]) for rp_idx in range(0,last_rp-1) if int(rp_list[rp_idx]-pre_s)>0]
                rpeaks_list.append([input_list[idx],int(rp_list[-1]),int(rp_list[-1]-pre_s),int(1000)])
            else:
                rpeaks_list += [rpeaks.tolist()]
    
        return rpeaks_list 

    def sortByID(self):
        sorted_dataInfo = self.dataInfo.sort_values(['ecg_id'])
        self.dataInfo = sorted_dataInfo.reset_index(drop=True)
        
    def divideMultiClassToIndividualSegments():
        return 0
    
#%% simple dataset  
class PTBXL_DataSet(Dataset):
    """ 
    This is a Dataset class for 10-second data.
    Inputs:
        path
        target_list - list of ecg_IDs for the given set
        upSample - variable that sets if upsampling, downsampling or no special
                    sampling strategy is performed
        leads - list of leads to include
        transform - no implemented functionality
        target_transform - no implemented functionality
    """
    def __init__(self, path, target_list, upSample = False, leads= ['V5'],
                 transform=None, target_transform=None):
        # Settings
        # print("Init started")
        self.path = path
        self.upSample = upSample
        
        self.leads = leads
        leads_list = ['I','II','III','aVL','aVR','aVF',
                      'V1','V2','V3','V4','V5','V6']
        self.lead_dict = dict(zip(leads_list,range(0,12)))
        if self.leads == ['all']:
            self.leads = leads_list
        
        self.transform = transform
        self.target_transform = target_transform
        
        # Raw unbalanced information about the dataset
        self.Y_raw = pd.read_csv(self.path+'ptbxl_database.csv',
                                 index_col='ecg_id')
        self.fileNames = self.Y_raw['filename_lr'].copy()
        self.ecg_labels = target_list
        
        # Processed data information with balanced classes
        self.dataInfo = None
        # self.labelInfo = None       
        if upSample == 1:
            self.upSampleMinor()
        elif upSample == 0:
            self.downSampleMajor()
        else:
            self.noSampling()
            print('No up or downsamplling is performed')
        # print("end of init")
        
    def __len__(self):
        # return len(self.ecg_labels)
        return len(self.dataInfo)

    def __getitem__(self, idx):
        # Load raw signal data
        # X = [wfdb.rdsamp(self.path+self.fileNames.iloc[self.ecg_labels['ecg_id'].iloc[idx]])]

        X_full = [wfdb.rdsamp(self.path+self.dataInfo['filename_lr'].iloc[idx])]
        X_full = np.array([signal for signal, meta in X_full]) # shape: [1,1000,12]
         
        X = np.ndarray([len(self.leads),X_full.shape[1]])
        for i,key in enumerate(self.leads):
            X[i] = X_full[0,:,self.lead_dict[key]]
        
        # This works instead
        # X = X_full[:,:,self.lead_dict[self.leads[0]]]
        X = X.reshape([len(self.leads) , X.shape[1]])

        
        # label = self.ecg_labels['class'].iloc[idx,-1]
        label = self.dataInfo['class'].iloc[idx]
        
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"X": X, "label": label}
        return sample
    
    def downSampleMajor(self):
        # print("Downsampling majority class")
        # Clear the lists
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              self.fileNames.name, 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples=(sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        n_samples = min(target_shape)
        for c in range(len(target_shape)):           
            # make random idx list
            idx_rand = np.random.permutation(len(self.ecg_labels[c]))
            
            # Pick n_samples random samples to keep 
            samples_idx = [self.ecg_labels[c][i] for i in idx_rand[0:n_samples]]
            
            # Add that to the list
            df = pd.DataFrame(self.fileNames[samples_idx])
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(samples_idx))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)
                    
        # # Shuffle
        # random_idx = np.random.permutation(len(self.dataInfo))
        # self.dataInfo = self.dataInfo.iloc[random_idx]
        
    def upSampleMinor(self):
        # Clear the lists
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              self.fileNames.name, 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples = (sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        n_samples = max(target_shape)
        for c in range(len(target_shape)): 
            # make random idx list
            multi = int(np.ceil(n_samples / len(self.ecg_labels[c])))
            long_ecg_labels = []
            long_ecg_labels += self.ecg_labels[c]*multi  
            
            idx_rand = np.random.permutation(len(long_ecg_labels)) 
            # idx_rand = np.random.randint(low=0,high=len(self.ecg_labels[c]),size=self.n_samples)
            
            # Pick n_samples random samples to keep
            samples_idx = [long_ecg_labels[i] for i in idx_rand[0:n_samples]]
            # samples_idx = [self.ecg_labels[c][i] for i in idx_rand] # idx_rand[0:self.n_samples]]
            
            # Add that to the list
            df = pd.DataFrame(self.fileNames[samples_idx])
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(samples_idx))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)
                    
        # # Shuffle
        # random_idx = np.random.permutation(len(self.dataInfo))
        # self.dataInfo = self.dataInfo.iloc[random_idx]
    def noSampling(self):
        # Clear the lists
        self.dataInfo = pd.DataFrame(columns=(self.fileNames.index.name,
                                              self.fileNames.name, 'class'))

        # Find the number of samples for each class
        target_shape = [len(a) for a in self.ecg_labels]
        
        # n_samples = (sum(target_shape)-max(target_shape))/(len(target_shape)-1)
        # n_samples = max(target_shape)
        for c in range(len(target_shape)): 
            # make random idx list
            long_ecg_labels = []
            long_ecg_labels += self.ecg_labels[c]  
            
            # Add that to the list
            df = pd.DataFrame(self.fileNames[long_ecg_labels])
            df[self.fileNames.index.name] = df.index
            df['class'] = np.ones(len(long_ecg_labels))*c
            
            self.dataInfo = self.dataInfo.append(df,ignore_index = True)    
    def divideMultiClassToIndividualSegments():
        return 0