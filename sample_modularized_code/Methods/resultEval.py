# -*- coding: utf-8 -*-
"""
Last updated Jul 21 2021

For code readability the Result evaluation methods
are implemented in this separate file

The methods defened in this file are:
    learningCurve()
    confMatrix()
    acc_measures()
    roc_auc()
    count_parameters()
    compare_roc()
    
@author: Balint
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
from prettytable import PrettyTable

from sklearn import metrics

np.random.seed(8)

def learningCurve(epoch, train_acc, valid_acc, figPath=''):
    """
    Method for plotting the learning curve
    
    Input: 
        epoch - list of epochs
        train_acc - list of training accuracies over each epoch
        valid_acc - list of validation accuracies over each epoch
        figPath - path to save the figure to
    Output:
        Returns nothing        
    """
    # plt.figure()  
    sn.axes_style("darkgrid")
    plt.rc('font', size=15)
    sn.set(font_scale=2, rc={"lines.linewidth": 2.5}) 
    
    # sn.despine(offset=10, trim=True)
    ratio = 0.03937007874015748 # conversion of mm to inch
    fig,ax = plt.subplots(figsize=[ratio*200,ratio*180])
    
    ax.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b', linewidth=3)
    plt.legend(['Train Acc', 'Val Acc'],loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim((0,1))
    
    plt.savefig(figPath+'learningCurve.png',bbox_inches='tight')
    plt.show()
    
def confMatrix(true_value, predicted, classDict, nr_classes=False,
               figPath ='confMatrix.png', norm=None, plotIt=1, myTitle=""):
    ''' 
    Method to extract confusion matrix and optionally plot it
    Input:
        true_value - list of true class labels
        predicted - list of predicted class labels
        classDict - dictionary mapping classes to numerical IDs
        nr_classes - optional, if sthg, this numberof rows will be in the plot
        figPath - path to save the figure to
        norm - if True the values in the conf matrix are normalized
        plotIt - if False, there is no plot generated
        myTitle - Takes the title for the plot
    Output:
        df_cm - a pandas dataframe of the confusion matrix
    '''
    if not nr_classes:
        nr_classes = len(classDict)
    confm = metrics.confusion_matrix(true_value, predicted, normalize=norm)
    df_cm = pd.DataFrame(confm, index=classDict, columns=classDict).T
    df_cm_norm = df_cm.copy()
    # print(df_cm_norm)
    df_cm_norm = df_cm_norm/df_cm.sum(axis=0)
    # print(df_cm_norm)
    # test_acc = metrics.balanced_accuracy_score(true_value, predicted)
    
    if plotIt:
        plt.rc('font', size=15)
        sn.set(font_scale=2)
        ratio = 0.03937007874015748 # conversion of mm to inch
        fig,ax = plt.subplots(figsize=[ratio*200,ratio*180])
        # plt.figure()        
        # colors = plt.cm.RdYlGn(norm(results.T.iloc[:,0]))
        sn.heatmap(df_cm_norm.iloc[0:nr_classes], cmap='RdYlGn',
                   annot=df_cm.iloc[0:nr_classes], fmt="1.0f",cbar=False,
                   annot_kws={"size": 50 / np.sqrt(len(df_cm_norm))})
        plt.ylabel('Predicted class')
        plt.xlabel('Actual class')
        # plt.title("Test set Acc:  %.4f" % test_acc)
        plt.savefig(figPath,bbox_inches='tight')
        plt.show()
        
    return df_cm

def acc_measures(confM, true_value, predicted,figPath='acc_measures.png',
                 plotIt=1,splitBars=1,title='.'):
    ''' 
    Method to extract accuracy measures and optionally plot it
    Input:
        confM - dataframe containing the confusion matrix
        true_value - list of true class labels
        predicted - list of predicted class labels
        figPath - path to save the figure to
        plotIt - if False, there is no barplot generated
        splitBars - if true, the bars for each perf. meas are splitted 
            according to the classes
        title - Takes the title for the plot
    Output: 
        results - a dataframe containing the performance measures as columns 
            and the classes as rows
    '''
    row_sum = confM.sum(axis=1)
    col_sum = confM.sum(axis=0)
    tp = np.diag(confM)
    
    # Acc = sum(np.diag(confM))/confM.sum().sum()
    Acc = metrics.balanced_accuracy_score(true_value, predicted)
    Sens = tp/row_sum
    Prec = tp/col_sum
    F1= 2*tp/(row_sum+col_sum)
    
    all_sum = row_sum.sum()
    tn = []
    for i in range(confM.shape[0]):
        for j in range(confM.shape[1]):
            if i == j:
                tn += [all_sum-row_sum[i]-col_sum[j]+confM.iloc[i,j]]
                
    Spec = tn/(all_sum-row_sum)
    results = pd.DataFrame({'Acc':Acc, 'Sens':Sens,'Spec':Spec,'Prec':Prec,'F1-score':F1})

    # print(results)
    results2 = results[['Sens','Spec','Prec','F1-score']].copy()
    results2['Class'] = confM.index
    results2 = results2.rename(index=dict(zip(confM.index,[my for my,_ in enumerate(confM.index)])))
    results2 = pd.melt(results2, id_vars='Class', var_name='Measures',
                       value_name='Value')
    # print(results2)
    if confM.shape[0] == 2:
        results = results.head(1)
        
    if plotIt:
        sn.axes_style("darkgrid")
        plt.rc('font', size=15)
        sn.set(font_scale=2, rc={"lines.linewidth": 2.5}) 
        sn.despine()
        
        norm = plt.Normalize(0.6, 0.95)
        colors = plt.cm.RdYlGn(norm(results.iloc[0,1:]))
        
        ratio = 0.03937007874015748 # conversion of mm to inch
        plt.figure(figsize=[ratio*200,ratio*180])  
        
        if splitBars:
            ax = sn.barplot(x='Measures',y='Value',hue='Class',
                            data=results2, palette=colors)
            plt.title(f'Accuracy: {Acc:.2f}')
        else:
            ax = sn.barplot(data=results[['Sens','Spec','Prec','F1-score']],
                            palette=colors)
            plt.title(f'Accuracy: {Acc:.2f}')
        
        # plt.title(label='Accuracy measures \n'+title,fontsize=18)
        ax.set(ylim=(0,1))
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                           (p.get_x() + p.get_width() / 2.,
                            p.get_height()/2), ha = 'center', va = 'center', 
                           xytext = (0, 9), textcoords = 'offset points',
                           size = 50 / np.sqrt(len(confM)))
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.savefig(figPath,bbox_inches='tight')
        plt.show()
    
    return results 

def roc_auc(true_value, predicted,figPath='roc.png', plotIt=1, title=''):
    ''' 
    Method to make ROC curve and calculate AUC and optionally plot it
    Inputs:
        true_value - list of true class labels
        predicted - list of predicted class labels
        figPath - path to save the figure to
        plotIt - if False, there is no barplot generated
        title - Takes the title for the plot
    Outputs:
        fpr,tpr,roc_auc - false positive rate, true positive rate and AUC
        otionally a plot
    '''
    fpr, tpr, thresholds = metrics.roc_curve(true_value, predicted)
    roc_auc = metrics.auc(fpr, tpr)
    if plotIt:
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='example estimator')
        display.plot()
        plt.title('ROC \n'+title)
        plt.savefig(figPath,bbox_inches='tight')
        plt.show()
    return fpr,tpr,roc_auc

def count_parameters(model):
    ''' 
    Method to count the number of trainable parameters in the model
    Inputs:
        model - pytorch model        
    Outputs:
        table,total_params - a table of the distribution of trainable
            parameters and a final count of them
    '''
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return table,total_params

def compare_roc(rocList,legend,figPath='roc.png'):
    ''' 
    Method to plot multiple ROC curves in one figure
    Inputs:
        rocList - list containing the data necessary for an 
            ROC curve (see output of roc_auc())
        legend - list of strings to put on the legend of the figure
        figPath - path to save the plot to
    Outputs:
        Returns a figure only
    '''
    sn.axes_style("darkgrid")
    plt.rc('font', size=15)
    sn.set(font_scale=2, rc={"lines.linewidth": 2.5}) 
    sn.despine(offset=10, trim=True)
    
    ratio = 0.03937007874015748 # conversion of mm to inch
    fig,ax = plt.subplots(figsize=[ratio*200,ratio*180])
        
    for i in range(len(rocList)):
        fpr = rocList[i][0]
        tpr = rocList[i][1]
        # ax = plt.plot(fpr,tpr)
        ax.plot(fpr,tpr,linewidth = 3)
    ax.set(xlim=(-0.01,1.01),
           ylim=(-0.01,1.01))

    # plt.title('ROC curves overlayed',fontsize='x-large')
    # plt.xlabel('False Positive Rate',fontsize='large')
    # plt.ylabel('True Positive Rate',fontsize='large') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  
    # plt.grid(True)
    # plt.legend(legend,loc='lower right',fontsize='large')
    plt.legend(legend,loc='lower right')
    plt.savefig(figPath,bbox_inches='tight')
    plt.show()
    return