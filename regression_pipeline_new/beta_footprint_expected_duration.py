#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% __________ [0] Imports
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import (
    tfr_array_morlet,
)

from mne.parallel import parallel_func

import pandas as pd

import os
import sys

from sklearn.metrics import roc_auc_score
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\scripts")
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\config")
import vst_config as config
import eeg_preprocessing as preprocessing


#%% __________ [1] Define directories
report_dir  = f'../reports/regression/'

subjects_list    = config.preprocessed_subjects_list[:1]             # list of participants (sid)
n_subjects       = len(subjects_list)             # number of participants

#_____  colors
Scmap = plt.cm.Greens
Tcmap = plt.cm.Purples
Scol  = [Scmap(100),Scmap(160), Scmap(220)]
Tcol  = [Tcmap(100),Tcmap(160), Tcmap(220)]

#____ frequencies of interest (FOI)
foi_names   = ['delta', 'theta','alpha','beta','Lgamma','Hgamma']
foi_list    = {'delta':   (1, 3),
               'theta':   (4, 7),  
               'alpha':   (8, 12), 
               'beta':    (14, 30), 
               'Lgamma':  (35,60), 
               'Hgamma':  (60, 119)}

#%%============================================================================
# --------------------------- YOUR SELECTION NEEDED ---------------------------
#==============================================================================

foi = 'beta'
epochs_type = "start_prod"
baseline_duration = 0.8
epoch_duration = 4
pick_type = "frontocentral"

t_min = int(700/4)
t_max = int(900/4)

# Number of standard deviations to define the categories of duration produced
n_std = 1

#%%============================================================================

# Takes the epochs as input and returns the power time course within given frequencies as an epochs array number of epochs x number of channels x times
def get_power_epochs(epochs, power_freq_low_bound, power_freq_high_bound):
    frequencies = np.arange(power_freq_low_bound, power_freq_high_bound)  
    cycles_per_freq = 6
    
    power = tfr_array_morlet(epochs.get_data(copy=False), freqs=frequencies, n_cycles=cycles_per_freq, output="power", sfreq=config.resample_sfreq, verbose=False)
    power_timecourse = np.mean(power, axis=2)

    # Transform to epochs and keep metadata
    epochs_power_timecourse = mne.EpochsArray(power_timecourse, info=epochs.info, verbose=False)
    epochs_power_timecourse.metadata = epochs.metadata

    return epochs_power_timecourse


#%%============================================================================

beta_footprints = []
duration_conditions = []

for i, subject in enumerate(subjects_list):

    #  Load, TFR transform and power time course of the epochs
    print(f'Processing: {subject}')
    for condition in config.conditions:

        condition_epochs = preprocessing.load_subject_epochs_by_type_and_condition_in_blocks(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)  
        
        for epochs in condition_epochs:
            # check if the epochs are empty
            if epochs.__len__() == 0:
                continue
            else:
                selected_channels = set(epochs.info['ch_names']).intersection(set(config.channels_from_cluster_analysis))
                epochs = epochs.pick(picks=list(selected_channels))
            power_epochs = get_power_epochs(epochs, foi_list[foi][0], foi_list[foi][1])
            beta_footprints.append(power_epochs.get_data(copy=False)[:,:,t_min:t_max].mean(axis = (1,2)))

            # Prepare the target
            if condition[-1] == "L":
                duration_conditions.append([1]*power_epochs.__len__())
            else:
                duration_conditions.append([0]*power_epochs.__len__())

# Linear REGRESSION
#____________________________________________________

# flatten beta footprints and produced durations and run a linear regression with scikit-learn
from sklearn.linear_model import LogisticRegression

beta_footprints = np.concatenate(beta_footprints)
duration_conditions = np.concatenate(duration_conditions)
X = beta_footprints.reshape(-1,1)
y = duration_conditions

#Shuffle data
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict_proba(X)[:, 1]

roc_auc = roc_auc_score(y, y_pred)

# Step 3: Plot the original data points and decision boundary
plt.scatter(X[y == 0], y[y == 0], color='blue', label='Short Duration')
plt.scatter(X[y == 1], y[y == 1], color='red', label='Long Duration')
plt.xlabel('Beta Footprint')
plt.ylabel('Label')

plt.title('Logistic Regression with AUC ROC: ' + str(roc_auc))
# Show legend
plt.legend()

plt.savefig(report_dir + 'beta_footprint_duration_condition_regression_frontocentral_plot.png')


            
        
        
    
    

# %%
