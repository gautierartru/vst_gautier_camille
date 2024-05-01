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
import eeg_decoding as decoding


#%% __________ [1] Define directories
report_dir  = f'../reports/decoding/'

subjects_list    = config.preprocessed_subjects_list[:12]                # list of participants (sid)
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
pick_type = "all_channels"

# Times of interest for the decoding
t_min = 0
t_max = 4

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

power_epochs_all = []
produced_durations = []

for i, subject in enumerate(subjects_list):

    #  Load, TFR transform and power time course of the epochs
    print(f'Processing: {subject}')
    for condition in config.conditions:
        condition_epochs = preprocessing.load_subject_epochs_by_type_and_condition_in_blocks(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)  
        # Loop over condition block
        for epochs in condition_epochs:
            # check if the epochs are empty
            if epochs.__len__() == 0:
                continue
            else:
                selected_channels = set(epochs.info['ch_names']).intersection(set(config.channels_from_cluster_analysis))
                epochs = epochs.pick(picks=list(selected_channels))
            power_epochs = get_power_epochs(epochs, foi_list[foi][0], foi_list[foi][1])
            power_epochs_all.append(power_epochs.crop(tmin=t_min, tmax=t_max))

            # Prepare the target
            produced_durations.append(produced_duration_category(epochs))
      
# DECODING
#____________________________________________________


#________________
# X matrix
concat_data = np.concatenate([power_epochs.get_data(copy=False) for power_epochs in power_epochs_all])
print(concat_data.shape)
    
# labels
y = np.concatenate(produced_durations)
print(y.shape)

#Shuffle data
from sklearn.utils import shuffle
concat_data,y = shuffle(concat_data,y, random_state=42)

classifier = "logistic"
scores = decoding.compute_decoding_scores(eeg_data=concat_data, labels=y, decoder=classifier, cv=4, ncores=-1)


fig, ax = decoding.plot_decoding_scores(
    decoding_scores=scores,
    epochs=power_epochs_all[0],
    plot_title = f"Sensor space decoding on {epochs_type} on {pick_type} epochs between Long and Short conditions",
    plotting_theme="ticks"
)

os.makedirs(report_dir, exist_ok=True)
plt.savefig(report_dir + f'beta_sliding_decoding_produced_duration_{pick_type}_plot.png')



# %%
