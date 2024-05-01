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
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\scripts")
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\config")
import vst_config as config
import eeg_preprocessing as preprocessing


#%% __________ [1] Define directories
report_dir  = f'../reports/regression/'

subjects_list    = config.preprocessed_subjects_list                # list of participants (sid)
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
expected_duration = "long"

t_min = 700
t_max = 900

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
# Separate the epochs by produced duration: shorter, normal, longer than the mean of the distribution of produced durations

def produced_duration_category(epochs):
    produced_duration = epochs.metadata["duration_produced"]
    
    shorter = produced_duration < (produced_duration.mean() - n_std * produced_duration.std())
    normal = (produced_duration >= (produced_duration.mean() - n_std * produced_duration.std())) & (produced_duration <= (produced_duration.mean() + n_std * produced_duration.std()))
    longer = produced_duration > (produced_duration.mean() + n_std * produced_duration.std())
    produced_duration_category = np.zeros(len(produced_duration))
    produced_duration_category[shorter] = -1
    produced_duration_category[normal] = 0
    produced_duration_category[longer] = 1
    return produced_duration_category

#%%============================================================================
# Z-score the produced durations
def produced_duration_z_score(epochs):
    produced_duration = epochs.metadata["duration_produced"]
    mean = produced_duration.mean()
    std_dev = produced_duration.std()

    # Calculate z-score
    produced_duration_zscored = (produced_duration - mean) / std_dev
    return produced_duration_zscored


#%%============================================================================

beta_footprints = []
produced_durations = []

for i, subject in enumerate(subjects_list):

    #  Load, TFR transform and power time course of the epochs
    print(f'Processing: {subject}')
    if expected_duration == "short":
        conditions = [condition for condition in config.conditions if condition[-1] == "S"]
    elif expected_duration == "long":
        conditions = [condition for condition in config.conditions if condition[-1] == "L"]

    for condition in conditions:

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
            produced_durations.append(produced_duration_z_score(epochs).values)

# Linear REGRESSION
#____________________________________________________

# flatten beta footprints and produced durations and run a linear regression with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

beta_footprints = np.concatenate(beta_footprints)
produced_durations = np.concatenate(produced_durations)

X = beta_footprints.reshape(-1,1)
y = produced_durations
print("Shape of beta:", beta_footprints.shape)
print("Shape of durations:", produced_durations.shape)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

plt.scatter(X, y, color='blue' )

# Step 4: Plot the linear regression line
plt.plot(X, y_pred, color='red', label='Linear regression')
r_squared = r2_score(y, y_pred)

# Add labels, legend, and title with R-squared value
plt.xlabel('Beta footprint')
plt.ylabel('Produced duration (z-score)')
plt.title(f'Linear Regression (R-squared = {r_squared:.4f})')
plt.legend()

os.makedirs(report_dir, exist_ok=True)

plt.savefig(report_dir + f'beta_footprint_relative_duration_regression_{t_min}_{t_max}___frontocentral_{expected_duration}_plot.png')


            
        
        
    
    

# %%
